import os


import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd


def spectrogram_from_eeg(parquet_path, display=False):
    
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')
    
    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
        
        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')
            
    if display: 
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print(); print('#'*25); print()
        
    return img


BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"


os.listdir(BASE_PATH)


NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

import pywt
print("The wavelet functions we can use:")
print(pywt.wavelist())

USE_WAVELET = None #or "db8" or anything below

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret

import librosa


data = spectrogram_from_eeg(BASE_PATH+"/train_eegs/"+os.listdir(BASE_PATH+"/train_eegs/")[0])
data.shape


def data_to_2d(data):
    arr1 = data[:, :, 0]
    arr2 = data[:, :, 0]
    arr3 = data[:, :, 0]
    arr4 = data[:, :, 0]
        # İlk iki array'i yatay olarak birleştir
    top_row = np.hstack((arr1, arr2))
    
    # Sonraki iki array'i yatay olarak birleştir
    bottom_row = np.hstack((arr3, arr4))
    
    # İki satırı dikey olarak birleştir
    result = np.vstack((top_row, bottom_row))
    
    # Sonuç boyutunu kontrol et
    # print(result.shape)  # (256, 512)
    return result


def np_array_from_eeg_id(eeg_id):
    data = spectrogram_from_eeg(BASE_PATH+"/train_eegs/"+str(eeg_id)+".parquet")
    return data_to_2d(data)


np_array_from_eeg_id(1628180742).shape


def visualize_eeg(eeg_id):
    
    plt.imshow(np_array_from_eeg_id(eeg_id), cmap='viridis', aspect='auto')  # İridis yerine Inferno paleti
    plt.colorbar()  # Renk barı ekle
    plt.title("Visualized Result Array with Iridis-like Palette")
    plt.show()


data_to_2d(data)


visualize_eeg(1628180742)


def save_spectrogram_as_img(path):
    #data =np.load("EEG_Spectrograms/1000913311.npy")
    data =np.load(path)
    # Concatenate the 4 pictures in a 2x2 grid
    concatenated_image = np.vstack((np.hstack((data[:, :, 0], data[:, :, 1])), 
                                    np.hstack((data[:, :, 2], data[:, :, 3]))))

    # # Display the concatenated image
    # plt.imshow(concatenated_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Save the concatenated image with the same name as the numpy file
    output_filename = path.split('/')[1].split('.')[0]
    output_filename = "images/"+output_filename+".jpg"
    cv2.imwrite(output_filename, concatenated_image * 255)  # Scale the image to 0-255 before saving


def get_img(id):
    data = spectrogram_from_eeg(BASE_PATH + "/train_eegs/" +  f"{id}.parquet")
    
    # Concatenate the 4 pictures in a 2x2 grid
    concatenated_image = np.vstack((
        np.hstack((data[:, :, 0], data[:, :, 1])),
        np.hstack((data[:, :, 2], data[:, :, 3]))
    ))
    
    # Normalize to 0-255 and convert to uint8
    normalized = cv2.normalize(concatenated_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_uint8 = normalized.astype(np.uint8)
    
    # Apply viridis colormap
    colored_img = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_VIRIDIS)
    
    return colored_img


get_img(1628180742)


csv = pd.read_csv(BASE_PATH+"/train.csv")


unique_eeg_ids_df = csv.drop_duplicates(subset='eeg_id')



unique_eeg_ids_df


unique_eeg_ids_df = unique_eeg_ids_df[['eeg_id', 'expert_consensus']]



unique_values = unique_eeg_ids_df['expert_consensus'].unique()
print(unique_values)


labels = {0:"Seizure",1:"GPD",2:"LRDA",3:"LPD",4:"GRDA",5:"Other"}


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Conv2D, BatchNormalization, Activation, Flatten, SpatialDropout2D
from tensorflow.keras.optimizers import Adam

# ResNet50V2 modelini yükle
base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 512, 3)))

# Base model'in ağırlıklarını dondur
base_model.trainable = False

# Yeni katmanları ekle
x = base_model.output

# Ekstra 2D konvolüsyon katmanları
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SpatialDropout2D(0.3)(x)

x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Küresel ortalama havuzlama
x = GlobalAveragePooling2D()(x)

# Tam bağlı katmanlar
dense_units = [512, 256]
for units in dense_units:
    x = Dense(units, activation='relu')(x)
    x = Dropout(0.5)(x)

# Çıkış katmanı (6 sınıf için)
output = Dense(6, activation='softmax')(x)

# Modeli oluştur
model = Model(inputs=base_model.input, outputs=output)

# Optimize edici
optimizer = Adam(learning_rate=0.001)

# Derleme
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Özet
model.summary()


def convert_to_rgb(image):
    # Tek kanal (grayscale) resmi 3 kanallı RGB'ye dönüştürme
    return tf.image.grayscale_to_rgb(image)


# Invert the labels dictionary to map string labels to their numeric values
label_map = {v: k for k, v in labels.items()}

# Replace the string values in the expert_consensus column with their numeric values
unique_eeg_ids_df['expert_consensus'] = unique_eeg_ids_df['expert_consensus'].map(label_map)
unique_eeg_ids_df


from sklearn.model_selection import train_test_split



# Önce eğitim ve geri kalan verileri (validasyon + test) ayıralım
train_df, rest_df = train_test_split(unique_eeg_ids_df, test_size=0.3, random_state=42) # %30 test + validasyon

# Geri kalan verileri validasyon ve test olarak ayıralım
val_df, test_df = train_test_split(rest_df, test_size=1/3, random_state=42) # %30'un 1/3'ü test, 2/3'ü validasyon


import keras.utils

class EEGDataGenerator(keras.utils.Sequence):
    """
    Data generator for EEG spectrograms for Keras.
    Converts EEG IDs to spectrograms using the provided function and returns batches.
    """
    
    def __init__(self, dataframe, spectrogram_function, batch_size=32, 
                 shuffle=True, seed=None, is_test=False):
        """
        Initialize the data generator.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'eeg_id' and 'expert_consensus' columns
            spectrogram_function (callable): Function that converts eeg_id to spectrogram array
            batch_size (int): Size of batches to generate
            shuffle (bool): Whether to shuffle the data after each epoch
            seed (int): Random seed for reproducibility
            is_test (bool): If True, don't return labels (for prediction)
        """
        self.df = dataframe.copy()
        self.batch_size = batch_size
        self.spectrogram_function = spectrogram_function
        self.shuffle = shuffle
        self.seed = seed
        self.is_test = is_test
        
        # Generate indices
        self.indices = np.arange(len(self.df))
        
        # Class mapping if needed
        self.classes = sorted(self.df['expert_consensus'].unique())
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        
        # Initial shuffle
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get batch data
        batch_df = self.df.iloc[batch_indices]
        
        # Generate spectrograms
        batch_x = np.array([
            self.spectrogram_function(eeg_id) 
            for eeg_id in batch_df['eeg_id']
        ])
        
        if self.is_test:
            return batch_x
        
        # Generate labels (one-hot encoded)
        batch_y = np.array([
            self.class_indices[label] 
            for label in batch_df['expert_consensus']
        ])
        
        return batch_x, tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)


def create_eeg_generators(train_df, val_df, test_df, spectrogram_from_eeg, 
                          batch_size=32, seed=42):
    """
    Create train, validation, and test generators for EEG data.
    
    Args:
        train_df (pd.DataFrame): Training data with 'eeg_id' and 'expert_consensus' columns
        val_df (pd.DataFrame): Validation data with 'eeg_id' and 'expert_consensus' columns
        test_df (pd.DataFrame): Test data with 'eeg_id' column
        spectrogram_from_eeg (callable): Function to convert eeg_id to spectrogram
        batch_size (int): Batch size for generators
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Create generators
    train_generator = EEGDataGenerator(
        dataframe=train_df,
        spectrogram_function=get_img,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        is_test=False
    )
    
    val_generator = EEGDataGenerator(
        dataframe=val_df,
        spectrogram_function=get_img,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        is_test=False
    )
    
    test_generator = EEGDataGenerator(
        dataframe=test_df,
        spectrogram_function=get_img,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        is_test=True
    )
    
    return train_generator, val_generator, test_generator


train_generator, val_generator, test_generator = create_eeg_generators(train_df, val_df, test_df, get_img, 
                          batch_size=32, seed=42)


from tensorflow.keras.optimizers import Adam

# Öğrenme oranını artır
learning_rate = 0.001  # Varsayılan 0.001, bunu artırdık

optimizer = Adam(learning_rate=learning_rate)

# Modeli yeniden derle
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)


model.save("resnet.h5")

