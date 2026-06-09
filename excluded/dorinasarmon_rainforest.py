!pip install torcheval


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
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics import MultilabelAccuracy
from glob import glob
import keras
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


seed=100
length=10
sr=48000
#slice_length=length*sr
image_height=128
image_width=400
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric=MultilabelAccuracy(threshold=0.5, criteria='hamming').to(device)
batch_size=16
epochs=30

save='/kaggle/working/spectrograms'
os.makedirs(save, exist_ok=True) # Kaggle-ben felesleges, amúgy is van working mappa


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


input='/kaggle/input/rfcx-species-audio-detection'
train_path=os.path.join(input, 'train')
test_path=os.path.join(input, 'test')
fp_csv=os.path.join(input, 'train_fp.csv')
tp_csv=os.path.join(input, 'train_tp.csv')
tp_df=pd.read_csv(tp_csv)
fp_df=pd.read_csv(fp_csv)


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

    filename=f'{species_id}_{recording_id}_{center}.bmp' # {center} kell, hátha ugyanolyan nevű file keletkezne
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
            image_width=image_width
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


# #fp_df=pd.read_csv(fp_csv)
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


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files=files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f=self.files[i]
        image=Image.open(f)
        image=image.convert('L') # grayscale
        image=np.array(image, dtype=np.float32)/255 # conv2d-hez
        image=torch.tensor(image)
        image=image.unsqueeze(0) # channel
        #image=image.unsqueeze(-1) # batch size
            
        label=int(os.path.basename(os.path.dirname(f)))
        s=torch.zeros(species)
        s[label]=1.0

        return image, s


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1, 16, (3,3)),
            nn.ReLU(),
            #nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (3,3)),
            nn.ReLU(),
            # nn.Conv2d(32, 64, (4,4)),
            # nn.ReLU(),
            #nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*(image_height-4)*(image_width-4), 128),
            nn.ReLU(),
            nn.Linear(128, species),
            nn.Sigmoid()
        )
       

    def forward(self, x):
        return self.model(x)


def train_model(
    model,
    training_loader,
    val_loader,
    device,
    epochs=30,
    lr=1e-3,
    patience=5,
    min_delta=1e-4,
    class_weights=None
):
    model.to(device)

    if class_weights is not None:
        loss_fn=nn.BCELoss(weight=class_weights)
        class_weights=class_weights.to(device)
    else:
        loss_fn=nn.BCELoss()

    optimizer=torch.optim.Adam(model.parameters())
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
    patience_count=0
    best_val_loss=float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss=0.0

        for inputs, labels in training_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*inputs.size(0) # batch_size
        train_loss/=len(training_loader.dataset)

        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs=model(inputs)
                metric.update(outputs, labels)
                loss=loss_fn(outputs, labels)
                val_loss+=loss.item()*inputs.size(0)

            val_loss/=len(val_loader.dataset)

        accuracy=metric.compute()
        metric.reset()
        scheduler.step(val_loss)
        print(f"Epoch: {epoch+1}/{epochs}")
        print(f"Accuracy: {accuracy}")
        print(f"Train loss: {train_loss}")
        print(f"Validation loss: {val_loss}")

        if val_loss<best_val_loss-min_delta:
            best_val_loss=val_loss
            patience_count=0
            best_model=model.state_dict()
        else:
            patience_count+=1
            if patience_count>=patience:
                print(f"Early stopping. Epoch: {epoch+1}")
                break

    model.load_state_dict(best_model)
    return model


all_files=glob("/kaggle/working/spectrograms/*/*.bmp")
print(f"Összes file: {len(all_files)}")
train_files, val_files=train_test_split(all_files, test_size=0.1)

train_dataset=AudioDataset(train_files)
val_dataset=AudioDataset(val_files)

training_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


model=CNN()
train_model(
    model=model,
    training_loader=training_loader,
    val_loader=val_loader,
    device=device,
    epochs=epochs
)


def gen_test(
    file_path,
    sr=48000,
    length=10,
    image_height=128,
    image_width=400):

    spectrograms=[]
    audio, _=librosa.load(file_path, sr=sr)
    slice_length=sr*length
    n=len(audio)//slice_length

    for i in range(n):
        start=i*slice_length
        end=start+slice_length
        if end>len(audio):
            end=len(audio)
        sliced_audio=audio[start:end]

        S = librosa.feature.melspectrogram(y=sliced_audio, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        S_norm = (S_norm * 255).astype(np.uint8)
        image=Image.fromarray(S_norm).resize((image_width, image_height))
        array=np.array(image)/255.0
        spectrograms.append(array)

    return spectrograms


def predict_test(
    model,
    spectrograms,
    device,
    threshold=0.5
):
    model.eval()
    model.to(device)
    inputs=[]

    for s in spectrograms:
        tensor=torch.tensor(s, dtype=torch.float32)
        tensor=tensor.unsqueeze(0).unsqueeze(0) #batch, channels
        inputs.append(tensor)
    inputs=torch.cat(inputs).to(device)

    with torch.no_grad():
        outputs=model(inputs)

    pred=outputs.max(dim=0).values
    binary_pred=(pred>threshold).int()

    return pred.cpu().numpy(), binary_pred.cpu().numpy()  


# file_path='/kaggle/input/rfcx-species-audio-detection/train/34340b225.flac'
# spectrograms=gen_test(file_path)
# pred, binary_pred=predict_test(model, spectrograms, device)

# print("Egyes fajok előfordulásának valószínűsége:")
# for i, probability in enumerate(pred):
#     print(f"{i}.\t{probability}")
# print("Binary prediction:")
# print(binary_pred)

# audio, sr=sf.read(file_path)
# ipd.Audio(audio, rate=sr)


def create_csv(model, test_path, device, csv_file=None):
    rows=[]
    
    for i, file in enumerate(sorted(os.listdir(test_path))):
        if file.endswith('.flac'):
            file_path=os.path.join(test_path, file)
            recording_id=file.replace('.flac', '')
            spectrograms=gen_test(file_path)
            pred, _=predict_test(model, spectrograms, device)
            #pred_rounded=[round(p, 1) for p in pred]
            pred_rounded=pred
            
            rows.append([recording_id]+list(pred_rounded))
        if i%100==0:
            print(f"{i} file feldolgozva.")
    
    df=pd.DataFrame(rows, columns=['recording_id']+[f"s{i}" for i in range(24)])
    if csv_file:
        df.to_csv(csv_file, index=False)
    else:
        print(df)
    


submission_dir='/kaggle/working/csv'
os.makedirs(submission_dir, exist_ok=True)
csv_file=os.path.join(submission_dir, 'submission.csv')
create_csv(model, test_path, device, csv_file=csv_file)

