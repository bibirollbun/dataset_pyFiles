import numpy as np
from tensorflow.keras.models import load_model


import os


os.listdir("/kaggle/input/modeller/keras/default/1/")


import shutil

shutil.copy(
    "/kaggle/input/modeller/keras/default/1/3d model 20nci epoch 67 17.h5",
    "/kaggle/working/3d.keras"
)




# Modelleri yükle
model_paths = [
    "/kaggle/input/modeller/keras/default/1/densenet 59nokta27.h5",
    "/kaggle/input/modeller/keras/default/1/inceptionresnetv2 57nokta69.h5",
    "/kaggle/input/modeller/keras/default/1/resnetv2 61nokta15.h5",
    "/kaggle/input/modeller/keras/default/1/3d model 20nci epoch 67 17.h5",
    "/kaggle/input/modeller/keras/default/1/63nokta25 ilkelmodel.h5",
    "/kaggle/input/modeller/keras/default/1/nasnet 60nokta62.h5"
]


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


cm = confusion_matrix(y_true, y_pred)


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')  # Renk skalası tercihe bağlı
plt.title("Confusion Matrix")
plt.show()



_3dmodel = load_model("/kaggle/working/3d.keras")


np.argmax(predict(_3dmodel,3372294787))


spectrogram_from_eeg(3372294787).shape


_3dmodel.predict(spectrogram_from_eeg(3372294787))


a = np.array([1, 2, 3])
b = np.array([3, 4, 5])

ortalama = (a + b) / 2
print(ortalama)  # [2. 3. 4.]


c


def calc_mean(eeg_id):
    toplam = None
    aray = spectrogram_from_eeg_2d(eeg_id)
    aray = np.expand_dims(aray, axis=0)        # (1, 512, 216)
    aray = np.expand_dims(aray, axis=-1) 
    aray = aray[:, :216, :, :]
    tahmin = _2dmodels[2].predict(aray)
    print("2 nolu modelin tahmini:",np.argmax(tahmin))
    print(tahmin)
    toplam = tahmin

    # Elindeki veri
    aray = np.random.rand(1, 216, 512)  # örnek olarak
    
    # Ekseni doğru sırala: (1, 512, 216)
    aray = np.transpose(aray, (0, 2, 1))
    
    # Son ekseni ekle: (1, 512, 216, 1)
    aray = np.expand_dims(aray, axis=-1)
    

    tahmin = _2dmodels[0].predict(aray)
    print("0 nolu modelin tahmini:",np.argmax(tahmin))
    print(tahmin)
    toplam +=tahmin
    tahmin = _2dmodels[1].predict(aray)
    print("1 nolu modelin tahmini:",np.argmax(tahmin))
    print(tahmin)
    toplam += tahmin
    
    tahmin = _2dmodels[3].predict(aray)
    print("3 nolu modelin tahmini:",np.argmax(tahmin))
    print(tahmin)
    toplam += tahmin
    tahmin = predict(_3dmodel,eeg_id)
    print("3d modelin tahmini:",np.argmax(tahmin))
    print(tahmin)
    toplam+=tahmin
    return toplam / (len(_2dmodels)+1)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def ensemble_predict(eeg_id):
    toplam = 0

    # 2D model (2. model) - özel preprocess
    aray = spectrogram_from_eeg_2d(eeg_id)
    aray = np.expand_dims(aray, axis=0)  # (1, 512, 216)
    aray = np.expand_dims(aray, axis=-1)
    aray = aray[:, :216, :, :]
    toplam += _2dmodels[2].predict(aray)

    # Diğer 2D modeller (0, 1, 3) - farklı formatta preprocess
    aray = spectrogram_from_eeg_2d(eeg_id)  # orijinal veri
    aray = np.transpose(aray, (1, 0))       # (216, 512)
    aray = np.expand_dims(aray, axis=0)     # (1, 216, 512)
    aray = np.expand_dims(aray, axis=-1)    # (1, 216, 512, 1)

    for i in [0, 1, 3]:
        toplam += _2dmodels[i].predict(aray)

    # 3D model
    aray = spectrogram_from_eeg(eeg_id)     # (128, 256, 4)
    aray = np.expand_dims(aray, axis=0)     # (1, 128, 256, 4)
    aray = np.expand_dims(aray, axis=-1)    # (1, 128, 256, 4, 1)
    toplam += _3dmodel.predict(aray)

    return toplam / (len(_2dmodels) + 1)

# Tüm test verisi için tahmin yap
y_true = []
y_pred = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    eeg_id = row['eeg_id']
    true_label = row['expert_consensus']

    probs = ensemble_predict(eeg_id)
    predicted_label = np.argmax(probs)

    y_true.append(true_label)
    y_pred.append(predicted_label)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Confusion Matrix
class_names = ["Seizure", "GPD", "LRDA", "LPD", "GRDA", "Other"]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Ensemble Confusion Matrix")
plt.show()



test_df


calc_mean(3372294787)


real = []
predictions = []





test_df





    for idx, row in ornekler.iterrows():
        eeg_id = row['eeg_id']
        dogru_label = row['expert_consensus']


len(test_df)



    ornekler = test_df.sample(50, random_state=42)  # aynı sonuç için sabit seed kullanılabilir

    count = 0
    for idx, row in test_df.iterrows():
        eeg_id = row['eeg_id']
        dogru_label = row['expert_consensus']

        tahmin = predict(_3dmodel,eeg_id)
        tahmin_edilen_label = np.argmax(tahmin)
        real.append(dogru_label)
        predictions.append(tahmin_edilen_label)
        count += 1
        if (count %25 == 0):
            print(count,",",end="")


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm  # İlerleme çubuğu için

# 1. Tüm spectrogram'ları yükle
def prepare_batch(test_df):
    X = []
    y = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        eeg_id = row['eeg_id']
        label = row['expert_consensus']
        
        spectro = spectrogram_from_eeg(eeg_id)
        spectro = np.expand_dims(spectro, axis=-1)      # (128, 256, 4, 1)
        X.append(spectro)
        y.append(label)
    
    X = np.array(X)                      # (N, 128, 256, 4, 1)
    y = np.array(y)                      # (N,)
    return X, y

# 2. Verileri hazırla
X_test, y_true = prepare_batch(test_df)




# 3. Toplu tahmin yap
y_probs = _3dmodel.predict(X_test, batch_size=16, verbose=1)
y_pred = np.argmax(y_probs, axis=1)

# 4. Confusion matrix çiz
class_names = ["Seizure", "GPD", "LRDA", "LPD", "GRDA", "Other"]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()


y_true


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=class_names))


# 3. Toplu tahmin yap
y_probs = _3dmodel.predict(X_test, batch_size=16, verbose=1)
y_pred = np.argmax(y_probs, axis=1)

# 4. Confusion matrix çiz
class_names = ["Seizure", "GPD", "LRDA", "LPD", "GRDA", "Other"]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # Virgülden sonra 4 basamaklı gösterim



cm = confusion_matrix(real, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')  # Renk skalası tercihe bağlı
plt.title("Confusion Matrix")
plt.show()



    # dogru_sayisi = 0
    # toplam = 0

    # ornekler = test_df.sample(50, random_state=42)  # aynı sonuç için sabit seed kullanılabilir

    # for idx, row in ornekler.iterrows():
    #     eeg_id = row['eeg_id']
    #     dogru_label = row['expert_consensus']

    #     tahmin = calc_mean(eeg_id)  # 6 elemanlı numpy array bekleniyor
    #     tahmin_edilen_label = np.argmax(tahmin)

    #     if tahmin_edilen_label == dogru_label:
    #         dogru_sayisi += 1

    #     print(tahmin)
    #     print("Tahmin:", tahmin_edilen_label, " Gerçek:", dogru_label, tahmin_edilen_label == dogru_label)

    #     toplam += 1
    #     dogruluk = dogru_sayisi / toplam
    #     print(f"Doğruluk: {dogruluk:.2%}")


test_df








def predict(model,eeg_id):
    aray = spectrogram_from_eeg(eeg_id)
    aray.shape
    aray = np.expand_dims(aray, axis=0)     # (1, 128, 256, 4)
    aray = np.expand_dims(aray, axis=-1)    # (1, 128, 256, 4, 1)
    return model.predict(aray)


def predict_2d(model,eeg_id):
    aray = spectrogram_from_eeg_2d(eeg_id)
    aray = np.expand_dims(aray, axis=0)        # (1, 512, 216)
    aray = np.expand_dims(aray, axis=-1) 
    aray = aray[:, :216, :, :]
    return model.predict(aray)


predict(_3dmodel,3260319360	)


_3dmodel.predict(test_df)


models[0]


models[0].predict(spectrogram_from_eeg(523993542))


spectrogram_from_eeg(523993542).shape


spectrogram_from_eeg_2d(523993542).shape


models



import shutil

shutil.copy(
    "/kaggle/input/modeller/keras/default/1/inceptionresnetv2 57nokta69.h5",
    "/kaggle/working/inceptionresnetv2 57nokta69.keras"
)


load_model("/kaggle/working/3d.keras")


models = []

for p in model_paths:
    print(p)
    try:
        models.append(load_model(p))
    except:
        print("hata")



load_model("/kaggle/input/modeller/keras/default/1/densenet 59nokta27.h5")


predictions = [model.predict(test_df) for model in models]



# Softmax çıktılarının ortalamasını al

average_prediction = np.mean(predictions, axis=0)





# En yüksek olasılığı olan sınıfı al
final_predictions = np.argmax(average_prediction, axis=1)


BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"



eeg_path = BASE_PATH+"/"+"train_eegs"



import pandas as pd



pd.read_parquet(eeg_path+"/"+os.listdir(eeg_path)[0])



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
import numpy as np


def spectrogram_from_eeg_2d(eeg_id, display=False):
    data = spectrogram_from_eeg(eeg_id)
    concatenated_image = np.vstack((np.hstack((data[:, :, 0], data[:, :, 1])), 
                                    np.hstack((data[:, :, 2], data[:, :, 3]))))
    return concatenated_image
    


def spectrogram_from_eeg(parquet_path, display=False):
    parquet_path = BASE_PATH+"/train_eegs/"+str(parquet_path)+".parquet"
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


NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]


csv = pd.read_csv(BASE_PATH+"/train.csv")



unique_eeg_ids_df = csv.drop_duplicates(subset='eeg_id')



unique_eeg_ids_df = unique_eeg_ids_df[['eeg_id', 'expert_consensus']]



unique_values = unique_eeg_ids_df['expert_consensus'].unique()
print(unique_values)


labels = {0:"Seizure",1:"GPD",2:"LRDA",3:"LPD",4:"GRDA",5:"Other"}



from tensorflow.keras import layers, models



# Invert the labels dictionary to map string labels to their numeric values
label_map = {v: k for k, v in labels.items()}

# Replace the string values in the expert_consensus column with their numeric values
unique_eeg_ids_df['expert_consensus'] = unique_eeg_ids_df['expert_consensus'].map(label_map)
unique_eeg_ids_df


from sklearn.model_selection import train_test_split



# Önce eğitim ve geri kalan verileri (validasyon + test) ayıralım
train_df, rest_df = train_test_split(unique_eeg_ids_df, test_size=0.2, random_state=42) # %30 test + validasyon

# Geri kalan verileri validasyon ve test olarak ayıralım
val_df, test_df = train_test_split(rest_df, test_size=1/2, random_state=42) # %30'un 1/3'ü test, 2/3'ü validasyon


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
        
        # Load spectrograms as NumPy arrays from .npy files
        batch_x = np.array([
            np.load(f"/kaggle/working/3d_images/{eeg_id}.npy") 
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
        spectrogram_function=spectrogram_from_eeg,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        is_test=False
    )
    
    val_generator = EEGDataGenerator(
        dataframe=val_df,
        spectrogram_function=spectrogram_from_eeg,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        is_test=False
    )
    
    test_generator = EEGDataGenerator(
        dataframe=test_df,
        spectrogram_function=spectrogram_from_eeg,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        is_test=True
    )
    
    return train_generator, val_generator, test_generator


BATCH_SIZE = 16
LEARNING_RATE = 0.001


train_generator, val_generator, test_generator = create_eeg_generators(train_df, val_df, test_df, spectrogram_from_eeg, 
                          batch_size=BATCH_SIZE, seed=42)


from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

