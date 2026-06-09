import numpy as np
import pandas as pd
import os
import shutil
import csv
import librosa
import librosa.display
import random
import matplotlib.pyplot as plt
import IPython.display as ipd
from PIL import Image
import soundfile as sf
import warnings
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


seed=100
length=5
sr=48000
image_height=128
image_width=400
batch_size=16
epochs=20

save='/kaggle/working/spectrograms'
os.makedirs(save, exist_ok=True)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


input='/kaggle/input/rfcx-species-audio-detection'
train_path=os.path.join(input, 'train')
test_path=os.path.join(input, 'test')
fp_csv=os.path.join(input, 'train_fp.csv')
tp_csv=os.path.join(input, 'train_tp.csv')
tp_df=pd.read_csv(tp_csv)
fp_df=pd.read_csv(fp_csv)


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


# flac_files=[f for f in os.listdir(train_path) if f.endswith('.flac')]
# random_file=random.choice(flac_files)
# random_file_path=os.path.join(train_path, random_file)
# print("flac_files hossza: ", len(flac_files))
# print('\nHang: ', random_file)
# recording_id=random_file.replace('.flac', '')
# print("Id:", recording_id)
# record=tp_df[tp_df['recording_id']==recording_id]
# y, sr=sf.read(random_file_path) # sr - sampling rate of y

# print(f"Sample rate: {sr}")
# if len(record)==0:
#     print("False positive.")
# else:
#     for _, row in record.iterrows():
#         print(f"Faj: {row['species_id']}")
#         print(f"Típus: {row['songtype_id']}")
#         print(f"Időtartam: {row['t_min']} - {row['t_max']}")
#         print(f"Frekvencia: {row['f_min']} - {row['f_max']}\n")
    
# ipd.Audio(y, rate=sr)


# S=librosa.feature.melspectrogram(y=y, sr=sr)
# fig, ax=plt.subplots()
# S_db=librosa.power_to_db(S, ref=np.max)
# image=librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
# fig.colorbar(image, ax=ax)
# ax.set(title='Spectrogram')


def spectrogram_gen(
    file_path,
    save,
    recording_id,
    species_id,
    time_min=None,
    time_max=None,
    sr=48000,
    length=10,
    image_height=128,
    image_width=400,
):
    slice_length=length*sr
    audio, _=librosa.load(file_path, sr=sr)

    center=(time_min+time_max)/2*sr
    start=max(center-slice_length//2, 0)
    end=start+slice_length
    if end>len(audio):
        end=len(audio)
        start=end-slice_length
    sliced_audio=audio[int(start):int(end)]

    S=librosa.feature.melspectrogram(y=sliced_audio, sr=sr)
    S_db=librosa.power_to_db(S, ref=np.max)
    S_norm=(S_db-S_db.min())/(S_db.max()-S_db.min())
    S_norm=(S_norm*255).astype(np.uint8)
    S_image=Image.fromarray(S_norm)
    S_image=S_image.resize((image_width, image_height))

    species_path=os.path.join(save, species_id)
    os.makedirs(species_path, exist_ok=True)

    filename=f'{species_id}_{recording_id}_{center}.png' # {center} kell, hátha ugyanolyan nevű file keletkezne
    save_path=os.path.join(species_path, filename)
    S_image.save(save_path)
    
    return save_path # későbbi visszanézésre


#tp_df=pd.read_csv(tp_csv)
ufiles=tp_df['recording_id'].nunique()
print(f"True Positive - fájlok száma (egyedi): {ufiles}")
print(f"True Positive - fájlok száma (összes): {len(tp_df)}")


with open(tp_csv) as f:
    reader=csv.reader(f)
    next(reader) # fejlécet átugorjuk
    for i, row in enumerate(reader):
        recording_id=row[0]
        species_id=row[1]
        time_min=float(row[3])
        time_max=float(row[5])
        file_path=os.path.join(train_path, recording_id + '.flac')
        audio, _=librosa.load(file_path, sr=sr)

        spectrogram_gen(
            file_path=file_path,
            save=save,
            recording_id=recording_id,
            species_id=species_id,
            time_min=time_min,
            time_max=time_max,
            sr=sr,
            length=length,
            image_height=image_height,
            image_width=image_width,
        )

        if i%100==0:
            print(f'{i} file feldolgozva.')


species=len([f for f in os.listdir(save) if os.path.isdir(os.path.join(save, f))])
print(f"Fajok száma: {species}")

sum_files=0
print("Fájlok száma az egyes species mappákban:")
for f in os.listdir(save):
    path=os.path.join(save, f)
    if os.path.isdir(path):
        files=len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        sum_files+=files
        print(f"{f}:\t{files}")
print(f"Összes file: {sum_files}")


# # Mappa törlése
# if os.path.exists(save):
#     shutil.rmtree(save)
#     print("Mappa törölve.")


#fp_df=pd.read_csv(fp_csv)
# ufiles=fp_df['recording_id'].nunique()
# print(f"False Positive - fájlok száma (egyedi): {ufiles}")
# print(f"False Positive - fájlok száma (összes): {len(fp_df)}")


# with open(fp_csv) as f:
#     reader=csv.reader(f)
#     next(reader) # fejlécet átugorjuk
#     for i, row in enumerate(reader):
#         recording_id=row[0]
#         species_id=row[1]
#         time_min=float(row[3])
#         time_max=float(row[5])
#         file_path=os.path.join(train_path, recording_id + '.flac')
#         audio, _=librosa.load(file_path, sr=sr)

#         spectrogram_gen(
#             file_path=file_path,
#             save=save,
#             recording_id=recording_id,
#             species_id=species_id,
#             time_min=time_min,
#             time_max=time_max,
#             sr=sr,
#             length=length,
#             image_height=image_height,
#             image_width=image_width
#         )

#         if i%100==0:
#             print(f'{i} file feldolgozva.')


# species=len([f for f in os.listdir(save) if os.path.isdir(os.path.join(save, f))])
# print(f"Fajok száma: {species}")

# print("Fájlok száma az egyes species mappákban:")
# for f in os.listdir(save):
#     path=os.path.join(save, f)
#     if os.path.isdir(path):
#         files=len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
#         print(f"{f}:\t{files}")


train_ds=tf.keras.utils.image_dataset_from_directory(
    directory=save,
    labels="inferred",
    label_mode="categorical",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_height, image_width),
    seed=seed,
    validation_split=0.1,
    subset="training"
)

val_ds=tf.keras.utils.image_dataset_from_directory(
    directory=save,
    labels="inferred",
    label_mode="categorical",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_height, image_width),
    seed=seed,
    validation_split=0.1,
    subset="validation"
)


class_labels=[]
for _, labels in train_ds:
    labels=tf.argmax(labels, axis=1)
    class_labels.extend(labels.numpy().tolist())
class_labels=np.array(class_labels)

class_weights=compute_class_weight(
    class_weight='balanced',
    classes=np.arange(species),
    y=class_labels
)

print(class_weights)

class_weights_dict=dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}")


# for images, _ in train_ds.take(1):
#     plt.figure(figsize=(15, 10))
#     for i in range(len(images)):
#         plt.subplot(4, 4, i+1)
#         plt.imshow(images[i])
#         plt.axis('off')
#     plt.suptitle("Első batch képei", fontsize=15)
#     plt.tight_layout()
#     plt.show()


model=Sequential([
    Input(shape=(image_height, image_width, 1)),
    #data_augmentation,
    layers.Rescaling(1./255),
    Conv2D(16, (3, 3), activation='relu'),
    #MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    #MaxPooling2D(pool_size=(2, 2)),
    #Conv2D(64, (3, 3), activation='relu'),

    Flatten(),
    #Dense(128, activation='relu'),
    #Dropout(0.25, seed=seed),
    Dense(128, activation='relu'),
    #Dropout(0.25, seed=seed),
    Dense(species, activation='sigmoid')
])


optimizer=Adam(
    #learning_rate=0.01
)

reduce_lr=ReduceLROnPlateau(
    factor=0.5,
    patience=5,
    verbose=1
)

early_stop=EarlyStopping(
    patience=5,
    verbose=1,
    restore_best_weights=True
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()


history=model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr]
)


# plt.figure(figsize=(10, 5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label='Training accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower center')

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='Train loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()


def gen_test_slice(
    file_path,
    sr=48000,
    length=10,
    image_height=128,
    image_width=400):
    
    spectrograms = []
    audio, _ = librosa.load(file_path, sr=sr)
    slice_length = sr * length
    n = len(audio) // slice_length

    for i in range(n):
        start = i * slice_length
        end = start + slice_length
        if end > len(audio):
            end = len(audio)
        sliced_audio = audio[start:end]

        S = librosa.feature.melspectrogram(y=sliced_audio, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        S_norm = (S_norm * 255).astype(np.uint8)
        image = Image.fromarray(S_norm).resize((image_width, image_height))
        array = np.array(image) / 255.0
        spectrograms.append(array)

    return spectrograms


def predict_test_slice(
    model,
    spectrograms,
    threshold=0.5
):
    inputs = []
    for s in spectrograms:
        tensor = np.expand_dims(s, axis=(0, -1))  # 1, height, width, 1
        inputs.append(tensor)
    inputs = np.concatenate(inputs, axis=0)

    outputs = model.predict(inputs, verbose=0)
    pred = np.max(outputs, axis=0)
    binary_pred = (pred > threshold).astype(int)
    return pred, binary_pred


def create_csv_slice(model, test_path, csv_file=None):
    rows = []

    for i, file in enumerate(sorted(os.listdir(test_path))):
        if file.endswith('.flac'):
            file_path = os.path.join(test_path, file)
            recording_id = file.replace('.flac', '')
            spectrograms = gen_test_slice(file_path)
            pred, _ = predict_test_slice(model, spectrograms)
            rows.append([recording_id] + list(pred))
        if i % 100 == 0:
            print(f"{i} file feldolgozva.")

    df = pd.DataFrame(rows, columns=['recording_id'] + [f"s{i}" for i in range(24)])
    if csv_file:
        df.to_csv(csv_file, index=False)
    else:
        print(df)



# def spectrogram_gen_test(
#     file_path,
#     image_size=(image_height, image_width),
#     normalize=True,
#     sr=48000
# ):
#     y, _=librosa.load(file_path, sr=sr)
#     S=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=image_size[0])
#     S=librosa.power_to_db(S, ref=np.max)
    
#     if normalize:
#         S=(S-S.min())/(S.max()-S.min())
        
#     S=tf.expand_dims(S, axis=-1)   
#     S=tf.image.resize(S, (image_height, image_width))
#     S=tf.expand_dims(S, axis=0) # batch
#     #print(S.shape)
#     return S


# def pred_image(
#     file_path,
#     image_size=(image_height, image_width),
#     normalize=True,
#     sr=48000,
#     threshold=0.5
# ):
#     input=spectrogram_gen_test(file_path, image_size, normalize, sr)
#     pred=model.predict(input, verbose=False)[0]
#     binary_pred=(pred>threshold).astype(int)
    
#     # print("Prediction:")
#     # print(pred)
#     # print("Binary prediction:")
#     # print(binary_pred)
#     # print(f"Pred:\t{pred.shape}")
#     # print(f"Binary_pred:\t{binary_pred.shape}")
#     # for i, prob in enumerate(pred):
#         # print(f"{i}\t{prob}")

#     # for i, prob in enumerate(pred):
#     #     print(f"{i}.\t{prob}")


# def create_csv(model, test_path, csv_file=None, 
#                                image_size=(image_height, image_width),
#                                normalize=True, sr=48000, threshold=0.5):
#     rows = []

#     for i, file in enumerate(sorted(os.listdir(test_path))):
#         if file.endswith('.flac'):
#             file_path = os.path.join(test_path, file)
#             recording_id = file.replace('.flac', '')
#             input_tensor = spectrogram_gen_test(file_path, image_size, normalize, sr)
#             pred = model.predict(input_tensor, verbose=0)[0]
#             binary_pred = (pred > threshold).astype(int)
#             rows.append([recording_id] + list(pred))

#         if i % 100 == 0:
#             print(f"{i} fájl feldolgozva.")

#     df = pd.DataFrame(rows, columns=['recording_id'] + [f's{i}' for i in range(24)])
#     if csv_file:
#         df.to_csv(csv_file, index=False)
#         print(f"Mentve: {csv_file}")
#     else:
#         print(df)


# file_path='/kaggle/input/rfcx-species-audio-detection/train/c12e0a62b.flac'
# gen_test_slice(file_path)
# pred_image(file_path)


submission_dir='/kaggle/working/csv'
os.makedirs(submission_dir, exist_ok=True)
csv_file=os.path.join(submission_dir, 'submission.csv')
create_csv_slice(model, test_path, csv_file=csv_file)

