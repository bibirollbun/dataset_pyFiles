import pandas as pd
import numpy as np
import tensorflow as tf
import os
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


def setup_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        device = 'TPU'
    except:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                if not tf.executing_eagerly():
                    tf.config.experimental.set_memory_growth(gpu, True)
            strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
            device = f'{len(gpus)} GPU(s)'
        else:
            strategy = tf.distribute.get_strategy()
            device = 'CPU'
    print(f'Using {device} with {strategy.num_replicas_in_sync} replicas')
    return strategy

strategy = setup_strategy()


with ZipFile('/kaggle/input/aerial-cactus-identification/train.zip') as zipper:
    zipper.extractall()
    
with ZipFile('/kaggle/input/aerial-cactus-identification/test.zip') as zipper:
    zipper.extractall()


labels = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
submission = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')
test_dir_df = pd.DataFrame({'id': os.listdir('/kaggle/working/test')})
train_dir_df = pd.DataFrame({'id': os.listdir('/kaggle/working/train')})


labels.sort_values('id').head()


train_dir_df.sort_values('id').head()


submission.sort_values('id').head()


test_dir_df.sort_values('id').head()


ee = set(test_dir_df['id']) == set(submission['id'])
le = len(test_dir_df) == len(submission)
print(f'elements are the same: {ee}')
print(f'lengths are equal: {le}')


num_train = len(train_dir_df)
num_test = len(test_dir_df)

print(f'number of train images: {num_train}')
print(f'number of test images: {num_test}')

cactus_count = labels['has_cactus'].sum()
no_cactus_count = len(labels) - cactus_count

print(f"number with cactus: {cactus_count} ({cactus_count/len(labels)})")
print(f"number without cactus: {no_cactus_count} ({no_cactus_count/len(labels)})")

sample_img = cv2.imread('/kaggle/working/train/' + labels['id'].iloc[9])
print(f"image shape: {sample_img.shape}")


plt.figure(figsize=(8, 6))
labels['has_cactus'].value_counts().plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title('Distribution of Classes')
plt.xlabel('Has Cactus')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Cactus (0)', 'Has Cactus (1)'], rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.show()


def display_aerial_images(labels=None, n_images=12, has_cactus=True):
    n_rows = int(np.floor(np.sqrt(n_images)))
    n_cols = int(np.ceil(n_images/n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()  

    if has_cactus:
        images = labels[labels['has_cactus'] == 1]['id'][-n_images:]
    else:
        images = labels[labels['has_cactus'] == 0]['id'][-n_images:]

    for idx, img_name in enumerate(images):
        img_path = 'train/' + img_name                 
        image = cv2.imread(img_path)                   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        axes[idx].imshow(image)
        axes[idx].axis('off') 
        
        for i in range(len(images), len(axes)):
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


display_aerial_images(labels=labels, n_images=20, has_cactus=True)


display_aerial_images(labels=labels, n_images=20, has_cactus=False)


# Create train and validation splits
train_df, val_df = train_test_split(labels, 
                                    test_size=0.2, 
                                    stratify=labels['has_cactus'], 
                                    random_state=19)

print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")


IMG_SIZE = 224
batch_size = 128

# used for training
datagen_augmentation = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# used for val and test
datagen_val_rescale = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen_test_rescale = ImageDataGenerator(preprocessing_function=preprocess_input)


# convert has_cactus to string for compatibility with flow_from_dataframe
train_df['has_cactus'] = train_df['has_cactus'].astype(str)
val_df['has_cactus'] = val_df['has_cactus'].astype(str)


train_generator = datagen_augmentation.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/working/train',
    x_col='id',
    y_col='has_cactus',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = datagen_val_rescale.flow_from_dataframe(
    dataframe=val_df,
    directory='/kaggle/working/train',
    x_col='id',
    y_col='has_cactus',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = datagen_test_rescale.flow_from_dataframe(
    dataframe=submission,
    directory='/kaggle/working/test',
    x_col='id',
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)


def plot_augmented_images(generator, title):

    x_batch, y_batch = next(generator)
    
    plt.figure(figsize=(16, 8))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(x_batch[i])
        plt.title(f"Class: {int(y_batch[i])}")
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


plot_augmented_images(train_generator, 'Augmented Training Images')


def build_model(freeze=True, n_trainable=10):

    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    input_layer = Input(shape=input_shape)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    # Freeze base model layers
    if freeze:
        for layer in base_model.layers:
            layer.trainable = False
    else:    
        for layer in base_model.layers[-n_trainable:]:
            layer.trainable = True
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    
    return model


model = build_model(freeze=False, n_trainable=10)
model.summary()


train_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)


# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_auc',
        mode='max',
        verbose=1
    )
]

steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight_dict
)


best_model = tf.keras.models.load_model('best_model.keras')


def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(16, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot AUC
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC (Area Under the ROC Curve)')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_training_history(history)


# get predictions on validation set
val_generator.reset()
Y_pred = model.predict(val_generator, steps=len(val_generator))
Y_pred_classes = (Y_pred > 0.5).astype(int)
Y_true = val_generator.classes

# confusion matrix
conf_matrix = confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Cactus', 'Has Cactus'],
            yticklabels=['No Cactus', 'Has Cactus'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# classification report
print("Classification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=['No Cactus', 'Has Cactus']))

# ROC curve
fpr, tpr, _ = roc_curve(Y_true, Y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


Y_true = np.array(Y_true)
print("Y_pred shape:", Y_pred.shape)
print("Y_true shape:", Y_true.shape)
print("Y_pred range:", np.min(Y_pred), np.max(Y_pred))

# check distribution of predictions
plt.figure(figsize=(8, 6))
plt.hist(Y_pred, bins=50)
plt.title('Distribution of Validation Predictions')
plt.show()

# checkch class balance in true labels
print("Class distribution in validation set:")
print(np.bincount(Y_true))

# ROC calculation
fpr, tpr, _ = roc_curve(Y_true, Y_pred)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc}")


# predict on test data
test_generator.reset()
test_predictions = model.predict(test_generator, steps=len(test_generator))

# Create submission dataframe
submission['has_cactus'] = test_predictions
submission.head()

# save file
submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")


def display_test_predictions(submission_df, n_images=12, threshold=0.5):
    n_rows = int(np.floor(np.sqrt(n_images)))
    n_cols = int(np.ceil(n_images/n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get a mix of predictions
    high_conf_cactus = submission_df[submission_df['has_cactus'] > 0.9]['id'].iloc[:n_images//3]
    low_conf_cactus = submission_df[(submission_df['has_cactus'] > threshold) & 
                                    (submission_df['has_cactus'] < 0.7)]['id'].iloc[:n_images//3]
    high_conf_no_cactus = submission_df[submission_df['has_cactus'] < 0.1]['id'].iloc[:n_images//3]
    
    images = pd.concat([high_conf_cactus, low_conf_cactus, high_conf_no_cactus])
    
    for idx, img_name in enumerate(images):
        if idx >= n_images:
            break
            
        img_path = 'test/' + img_name
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pred = submission_df.loc[submission_df['id'] == img_name, 'has_cactus'].values[0]
        
        axes[idx].imshow(image)
        axes[idx].set_title(f"Pred: {pred}")
        
        # Red border for predicted cactus, blue for no cactus
        border_color = 'red' if pred > threshold else 'blue'
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(7)
            spine.set_visible(True)
    
    for i in range(len(images), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

display_test_predictions(submission, n_images=16)


Y_pred_classes = (Y_pred > 0.5).astype(int)
print("Y_pred shape:", test_predictions.shape)

# check distribution of test predictions
plt.figure(figsize=(8, 6))
plt.hist(test_predictions, bins=50)
plt.title('Distribution of Test Predictions')
plt.show()

