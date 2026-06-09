#Data handling
import os
import pandas as pd
from PIL import Image
from glob import glob

# Randomization
import random

# Audio handling
import librosa

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Feedback with progress bar
from tqdm.notebook import tqdm

# Math & Algorithms
import numpy as np

# Model
import keras
from keras import layers
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.layers import Resizing

# Clustering
from sklearn.model_selection import train_test_split


# NN training parameters
segment_length = 0.8
batch_size = 32
lr = 1e-3
patience = 40
epochs = 1000

# Initialize random number generation
random_seed = 42
random.seed(random_seed)
rng = np.random.default_rng()


# Inputs
labeled_files_path='/kaggle/input/spectrograms-training-labeled-multiple'
test_path='/kaggle/input/rfcx-species-audio-detection/test'
autoencoder_filepath="/kaggle/input/model-38/keras/default/1/model_epoch_38.keras"

# Outputs
checkpoint_frozen_filepath="/kaggle/working/checkpoint_frozen/model_epoch_{epoch}.keras"
logger_frozen_filename="training_frozen.log"
checkpoint_full_filepath="/kaggle/working/checkpoint_full/model_epoch_{epoch}.keras"
logger_full_filename="training_full.log"
logger_combined="training_combined.log"
submission_dir = '/kaggle/working/csv'

# Create folders
os.makedirs("/kaggle/working/checkpoint_frozen", exist_ok=True)
os.makedirs("/kaggle/working/checkpoint_full", exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)


# Loading autoencoder
autoencoder=load_model(autoencoder_filepath)

# Getting encoder
encoder=autoencoder.get_layer("encoder")
encoder.summary()

# Getting latent dim
latent_dim=encoder.output_shape[-1]
print(latent_dim)


# Data augmentation
data_augmentation=keras.Sequential([
    layers.RandomTranslation(
        height_factor=0,
        width_factor=0.1,
        fill_mode='nearest'
    ),
    layers.RandomContrast(0.2),
    layers.RandomZoom(
        height_factor=0,
        width_factor=0.1
    )
])


class AudioDataset(keras.utils.Sequence):
    
    # Initialization
    def __init__(self, files, num_species, batch_size=16, shuffle=False, seed=None, **kwargs):
        """ Spectrogram dataset loader - reproducible
            
            Arguments:
            files: list of paths to images
            num_species: number of labels
            batch_size: number of files in a batch
            shuffle: whether to shuffle images
            seed: seed for reproducibility
        """
        super().__init__()      

        self.files=files.copy()
        self.num_species=num_species
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.seed=seed
        self.rng=np.random.RandomState(seed) if seed is not None else np.random
        
        # setting seeds
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            random.seed(seed)

        # detecting image shape
        with Image.open(self.files[0]) as image:
            self.input_shape=(image.size[1], image.size[0], 1)

        self.end_of_epoch()


    # Number of batches
    def __len__(self):
        return int(np.ceil(len(self.files)/self.batch_size))


    # Creates a single batch
    def __getitem__(self, index):
        batch_files=self.files[index*self.batch_size:(index+1)*self.batch_size]

        batch_images=[]
        batch_labels=[]
        
        for f in batch_files:
            with Image.open(f) as image:
                image=image.resize(self.input_shape[:2][::-1])
                image=np.array(image, dtype=np.float32)/255.0
                if image.ndim==2:
                    image=image[...,np.newaxis]

            
            batch_images.append(image)

            label=int(os.path.basename(os.path.dirname(f)))
            one_hot=np.zeros(self.num_species, dtype=np.float32)
            one_hot[label]=1.0
            batch_labels.append(one_hot)

        return np.stack(batch_images), np.stack(batch_labels)

    def get_all_items(self):
        data_images=[]
        data_labels=[]
        
        for f in self.files:
            with Image.open(f) as image:
                image=image.resize(self.input_shape[:2][::-1])
                image=np.array(image, dtype=np.float32)/255.0
                if image.ndim==2:
                    image=image[...,np.newaxis]
            data_images.append(image)

            label=int(os.path.basename(os.path.dirname(f)))
            one_hot=np.zeros(self.num_species, dtype=np.float32)
            one_hot[label]=1.0
            data_labels.append(one_hot)

        return np.stack(data_images), np.stack(data_labels)

    
    # Shuffles files
    def end_of_epoch(self):
        if self.shuffle:
            if self.seed is not None:
                permutation=self.rng.permutation(len(self.files))
                self.files=[self.files[i] for i in permutation]
            else:
                np.random.shuffle(self.files)
        

    # Shape of an image
    def image_shape(self):
        return self.input_shape


# All files
all_files = glob("/kaggle/input/spectrograms-training-labeled-multiple/*/*.png")
print(f"Total files: {len(all_files)}")

# Getting number of species
num_species=len([int(d) for d in os.listdir(labeled_files_path) if os.path.isdir(os.path.join(labeled_files_path, d))])
print(num_species)

# Class weights
class_weights = np.ones([num_species])
for i in range(num_species):
    dir_path = os.path.join(labeled_files_path, str(i))
    class_weights[i] = 1/len(os.listdir(dir_path))
class_weights /= np.max(class_weights)
weight_dict = {}
for i, value in enumerate(class_weights):
    weight_dict[i] = value
print(weight_dict)

# Separating training and validation data
train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=random_seed)
train_dataset=AudioDataset(files=train_files, num_species=num_species, batch_size=batch_size, shuffle=True, seed=random_seed)
val_dataset=AudioDataset(files=val_files, num_species=num_species, batch_size=batch_size, shuffle=False, seed=random_seed)


# Getting input shape
target_shape=train_dataset.image_shape()
print(target_shape)
print(f"Input shape: {target_shape}")
print(len(train_dataset))
print(len(val_dataset))


def make_classifier(latent_dim, num_species):
    classifier_input=layers.Input(shape=(latent_dim,))
    # x=layers.Dense(256, activation='relu')(classifier_input)
    # x=layers.Dropout(rate=0.3, seed=random_seed)(x)
    x=layers.Dense(128, activation='relu')(classifier_input)
    x=layers.Dropout(rate=0.2, seed=random_seed)(x)
    classifier_output=layers.Dense(num_species, activation='softmax')(x)
    # classifier_output=layers.Dense(num_species, activation='softmax')(classifier_input)
    classifier=keras.Model(classifier_input, classifier_output)
    return classifier


# Make classifier
classifier=make_classifier(latent_dim, num_species)
# Freeze encoder layers
encoder.trainable=False
# Setting up model
model_input=layers.Input(shape=target_shape)
latent=encoder(model_input)
model_output=classifier(latent)
model=keras.Model(model_input, model_output)


# Optimizer
optimizer=keras.optimizers.Adam(learning_rate = lr)

# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = patience / 2, verbose=1)
early_stop = keras.callbacks.EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True)
checkpoint=keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_frozen_filepath,
    verbose=1
)
csv_logger=keras.callbacks.CSVLogger(
    filename=logger_frozen_filename,
    append=True
)


# Compile model
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
encoder.summary(show_trainable=True)
model.summary(expand_nested=True)


history_frozen=model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    class_weight = weight_dict,
    # callbacks = [early_stop, reduce_lr, csv_logger, checkpoint],
    callbacks = [early_stop, reduce_lr, csv_logger],
    verbose=1
)


# Unfreeze encoder layers
encoder.trainable=True
encoder.summary(show_trainable=True)
# Setting lower learning rate
lr=1e-4

# Setting different paths
checkpoint=keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_full_filepath,
    verbose=1
)
csv_logger=keras.callbacks.CSVLogger(
    filename=logger_full_filename,
    append=True
)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history_full = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    class_weight = weight_dict,
    # callbacks=[early_stop, reduce_lr, checkpoint, csv_logger],
    callbacks=[early_stop, reduce_lr, csv_logger],
    verbose=1
)


# Read logger files
df_frozen=pd.read_csv(logger_frozen_filename)
df_full=pd.read_csv(logger_full_filename)

# Combine logger files
df_full["epoch"]=df_full["epoch"]+df_frozen["epoch"].max()+1
df_combined=pd.concat([df_frozen, df_full], ignore_index=True)
print(df_combined.head())

# Save combined file
df_combined.to_csv(logger_combined, index=False)


log=pd.read_csv(logger_combined)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(log['loss'], label='Training loss')
plt.plot(log['val_loss'], label='Validation loss')
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(log['accuracy'], label='Training accuracy')
plt.plot(log['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('/kaggle/working/loss_acc.png', bbox_inches='tight')
plt.show()


from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_curve, confusion_matrix

# Create predictions for the test images
test_data, test_labels = val_dataset.get_all_items()
test_confidences = model.predict(test_data, verbose=0)
test_labels = np.argmax(test_labels, 1)
test_preds = np.argmax(test_confidences, 1)

print("Test accuracy: %g" %(accuracy_score(test_labels, test_preds)))
print("Test loss:", log_loss(test_labels, test_confidences)) # Average loss, not sum
print("Test precision:", precision_score(test_labels, test_preds, average = "macro"))
print("Test recall:", recall_score(test_labels, test_preds, average="macro"))
print("Test f1_score:", f1_score(test_labels, test_preds, average="macro"))


conf = confusion_matrix(test_labels, test_preds)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect(1)

res = sns.heatmap(conf, annot=True, vmin=0.0, fmt='d', cmap = plt.get_cmap('Blues'))

plt.ylim([0, 24])
plt.ylabel('True label')
plt.xlim([0, 24])
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')

res.invert_yaxis()

plt.show()
plt.close()


# Saving the model
model.save("model.keras")


# Generate spectrograms from a given file
def gen_test_spectrograms(file_path, segment_length, target_shape):
    
    # Load audio
    audio, sr = librosa.core.load(file_path, sr=None)

    # Number of (full) segments
    spectrograms=[]
    sample_length=int(segment_length*sr)
    num_segments=len(audio)//sample_length

    for i in range(num_segments):
        # Calculating start and end of the segment
        start = i * sample_length
        end = start + sample_length
        audio_short = audio[start:end]
        
        # Creating spectrograms
        S = librosa.feature.melspectrogram(y=audio_short, sr=sr, n_mels=target_shape[0])
        S_db = librosa.power_to_db(S, ref=np.max)      

        if S_db.shape[1]>target_shape[1]:
            S_db=S_db[:, :target_shape[1]]
        elif S_db.shape[1]<target_shape[1]:
            pad_width=[(0, 0), (0, target_shape[1]-S_db.shape[1])]
            S_db=np.pad(S_db, pad_width=pad_width, mode='constant')

        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        spectrograms.append(S_norm[..., None])  # Channel dimension

    return spectrograms


# Prediction on test files
def predict_test(model, spectrograms, threshold=0.):
    inputs=np.array(spectrograms)
    outputs=model.predict(inputs, verbose=0)
    pred=np.max(outputs, axis=0)
    binary_pred=(pred>threshold).astype(int)
    return pred, binary_pred


# Creating .csv file for submission
def create_csv(model, test_path, segment_length, target_shape, csv_file=None):
    rows = []
    test_paths = os.listdir(test_path)

    for i in tqdm(range(len(test_paths))):
        file = test_paths[i]
        
        if file.endswith('.flac'):
            file_path = os.path.join(test_path, file)
            recording_id = file.replace('.flac', '')
            spectrograms = gen_test_spectrograms(file_path, segment_length=segment_length, target_shape=target_shape)
            pred, _ = predict_test(model, spectrograms)
            rows.append([recording_id] + list(pred))
            # _, binary_pred=predict_test(model_keras, spectrograms)
            # rows.append([recording_id]+list(binary_pred))

    df = pd.DataFrame(rows, columns=['recording_id']+[f"s{i}" for i in range(num_species)])
    if csv_file:
        df.to_csv(csv_file, float_format='%.5f', index=False)
    else:
        print(df)


# Saving submission file
csv_file = os.path.join(submission_dir, 'rainForest_submission_keras.csv')
create_csv(model, test_path, segment_length, target_shape, csv_file=csv_file)

