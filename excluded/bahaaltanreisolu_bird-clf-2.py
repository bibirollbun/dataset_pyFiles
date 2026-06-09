!pip install librosa


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from IPython.display import Audio
from IPython.core.display import display
import IPython.display as ipd

import torch
import torchaudio #audio preprocessing via gpu
import torch,torchvision
from torch import nn
from torch.nn import functional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,IterableDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
import timm
import os
from glob import glob
import sys
sys.path.append("/kaggle/input/silero_useful/pytorch/default/1/src/")


from silero_vad.utils_vad import get_speech_timestamps
from silero_vad.model import load_silero_vad
import librosa

import torchaudio.transforms as T
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class ConfigMein:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    authors= [
    'Alexandra Butrago-Cardona',
    'Ana María Ospina-Larrea | Daniela Murillo',
    'Diego A Gómez-Morales',
    'Eliana Barona- Cortés',
    'Eliana Barona-Cortés | Daniela García-Cobos',
    'Paula Caycedo-Rosales | Juan-Pablo López',
    'Fabio A. Sarria-S'
    ]


train_csv=pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
tax_csv=pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")
merged=pd.merge(train_csv,tax_csv,how="left",left_on=["primary_label","common_name"],right_on=["primary_label","common_name"])
prob_labels=pd.read_csv("/kaggle/input/birdclef-2025/sample_submission.csv").drop("row_id",axis=1)


prob_labels


dummi=pd.get_dummies(merged["primary_label"],dtype=float)
dummi


data=pd.concat([merged,dummi],axis=1)
data


torch.set_num_threads(1)
model=load_silero_vad()


class EDA(nn.Module):
    def __init__(self,df=merged):
        super().__init__()
        self.df=df
        self.authors=ConfigMein.authors
        self.label_f=data.drop(['primary_label','secondary_labels','type','collection','rating','url','latitude','longitude','scientific_name_x','common_name','author','license','inat_taxon_id','scientific_name_y','class_name'],axis=1)
    def human_voice_detector(self,wav,sr):
        

        print(wav.shape)
        # Calculate the sound power
        power = wav ** 2
        
        # Split the data into chunks and sum the energy in every chunk
        chunk = int(chunk_len * sr)
        
        pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
        power = np.pad(power, (0, pad))
        power = power.reshape((-1, chunk)).sum(axis=1)

        speech_timestamps = get_speech_timestamps(torch.Tensor(wav), model,threshold=0.4)
        segmentation = np.zeros_like(wav)
        for st in speech_timestamps:
            segmentation[st['start']: st['end']] = 20
    
        fig = plt.figure(figsize=(24, 3))
        fig.suptitle(f'{rec.filename} by {rec.author}')
        
        t = np.arange(len(power)) * chunk_len
        plt.plot(t, 10 * np.log10(power), 'b')
        
        t = np.arange(len(segmentation)) / sr
        plt.plot(t, segmentation, 'r')        
        plt.show()
        
        display(Audio(fname))
    def delete_human_sound(self,wav,sample_rate):
        
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True, threshold=0.4);
        prev_end=0.0
        part=[]
        for i in speech_timestamps:
            init,end=i["start"],i["end"]
    
            if(init-prev_end>0.5):
                part.append(wav[int(prev_end * sample_rate):int(init * sample_rate)])
            prev_end=end
        if prev_end*sample_rate < wav.shape[0]:
             part.append(wav[int(prev_end*sample_rate):])
    
        if len(part) > 0:
            filtered_wav= np.concatenate(part, axis=0)
        else:
            filtered_wav= wav
        return filtered_wav
    def delete_human_sound_torch(self,wav,sample_rate=16000):
        
        
        # Eğer stereo ise mono'ya çevir
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # Kanalları ortalayarak mono'ya çevir
        
        
        
        prev_end = 0.0
        part = []
        for i in speech_timestamps:
            init, end = i["start"], i["end"]
        
            # Eğer arada 0.4 saniyeden uzun boşluk varsa o kısmı kaydet
            if init - prev_end > 0.4:
                start_idx = int(prev_end * sample_rate)
                end_idx = int(init * sample_rate)
                part.append(wav[:, start_idx:end_idx])  # Doğru indeksleme
        
            prev_end = end
        
        # Eğer sondaki ses parçalanmamışsa, onu da ekle
        if int(prev_end * sample_rate) < wav.shape[1]:
            part.append(wav[:, int(prev_end * sample_rate):])
        
        # Parçaları birleştir
        if len(part) > 0:
            filtered_wav = torch.cat(part, dim=1)  # Doğru eksende birleştir
            
        else:
            filtered_wav = wav
            
        return filtered_wav    
    
    def sub_eda(self):
       for _,row in self.df.iterrows():
            link="/kaggle/input/birdclef-2025/train_audio/"+row["filename"]
            wav,sr=librosa.load(link,sr=16000)
            wav, index = librosa.effects.trim(wav,top_db=20)
            
            
            aut=row["author"]
            there=row["collection"]
            labels = self.label_f["/kaggle/input/birdclef-2025/train_audio/"+self.label_f.filename == link].drop("filename", axis=1).to_numpy().flatten()
    
            if (aut in self.authors) and there=="CSA":
                wav=self.delete_human_sound(wav,sr)
            else:
                pass            
            
            
            
            segments=self.segment_audio(wav,sr,5)
            for segment in segments:
                spectrogram = librosa.stft(segment, n_fft=320, hop_length=32, win_length=320)
                spectrogram = np.abs(spectrogram)
                spectrogram = np.expand_dims(spectrogram, axis=2)
                
                yield spectrogram,labels
    def sub_eda2(self,link):
       
        
        wav,sr=librosa.load(link,sr=16000)
        wav, index = librosa.effects.trim(wav,top_db=20)
            
        wav = wav / np.max(np.abs(wav))
        aut=self.df["/kaggle/input/birdclef-2025/train_audio/"+self.df.filename==link]["author"].item()
        there=self.df["/kaggle/input/birdclef-2025/train_audio/"+self.df.filename==link]["collection"].item()
        labels = self.label_f["/kaggle/input/birdclef-2025/train_audio/"+self.label_f.filename == link].drop("filename", axis=1).to_numpy().flatten()
    
        if (aut in self.authors) and there=="CSA":
            wav=self.delete_human_sound(wav,sr)
        else:
            pass            
              
        segment=self.segment_audio2(wav,sr,5) 
        spectrogram = librosa.stft(segment, n_fft=320, hop_length=32, win_length=320)
        spectrogram = np.transpose(np.abs(spectrogram))
        spectrogram = torch.tensor(np.expand_dims(spectrogram, axis=0))
        
        return spectrogram,labels
        
    def for_test(self,links,mode):
        if(mode=="Testfor"):
            for link in links:
                name=(link[45:])[:-4]
                wav,sr=librosa.load(link,sr=16000)
                wav, index = librosa.effects.trim(wav,top_db=20)
                #wav=self.delete_human_sound(wav,sr)
                segments=self.segment_audio(wav,sr,5) 
                for idx , segment in enumerate(segments):
                    num=(idx+1)*5
                    spectrogram = librosa.stft(segment, n_fft=320, hop_length=32, win_length=320)
                    spectrogram = np.transpose(np.abs(spectrogram))
                    spectrogram = torch.tensor(np.expand_dims(spectrogram, axis=0))
                    index=f"{name}_{num}"
                    yield spectrogram,index
        else:
            for link in links:
                name=(link[46:])[:-4]
                wav,sr=librosa.load(link,sr=16000)
                wav, index = librosa.effects.trim(wav,top_db=20)
                #wav=self.delete_human_sound(wav,sr)
                segments=self.segment_audio(wav,sr,5) 
                for idx , segment in enumerate(segments):
                    num=(idx+1)*5
                    spectrogram = librosa.stft(segment, n_fft=320, hop_length=32, win_length=320)
                    spectrogram = np.transpose(np.abs(spectrogram))
                    spectrogram = torch.tensor(np.expand_dims(spectrogram, axis=0))
                    index=f"{name}_{num}"
                    yield spectrogram,index            
            
    def sub_eda_torch(self,link):
        new_sr=16000
        wav,sr=torchaudio.load(link)
        wav=wav.to("cuda")
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=new_sr)
        speech_timestamps = get_speech_timestamps(wav.squeeze(0), self.model, return_seconds=True, threshold=0.4)
        aut=self.df["/kaggle/input/birdclef-2025/train_audio/"+self.df.filename==link]["author"].item()
        if aut in self.authors:
            wav=self.delete_human_sound_torch(wav,new_sr).squeeze()
            
        else:
            wav=wav.squeeze()
        wav = torchaudio.functional.vad(wav, sample_rate=new_sr,trigger_level=0.1)
        
        wav = wav / torch.max(torch.abs(wav))
        
        """mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)"""
        mfcc_transform = T.MFCC(
            sample_rate=sr, 
            n_mfcc=40, 
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "center": False}
        ).to("cuda")
        
        mfcc = mfcc_transform(wav)

        mfcc_processed = torch.mean(mfcc, dim=-1)
        
        return mfcc_processed
    def segment_audio(self,wav, sr, segment_duration=5):
        samples_per_segment = segment_duration * sr  # 5 saniye = 5 * sr örnek
        segments=[]
        wav=np.pad(wav,((12*samples_per_segment)-len(wav),0))
        for i in range(0, len(wav), samples_per_segment):
            segment = wav[i:i + samples_per_segment]
            
            # Eğer segment 5 saniyeden kısa ise, sıfırlarla doldur
            if len(segment) < samples_per_segment:
                segment = np.pad(segment, (samples_per_segment - len(segment),0))
            
            segments.append(segment)
        return segments
        
    def segment_audio2(self,wav, sr, segment_duration=5):
        samples_per_segment = segment_duration * sr  # 5 saniye = 5 * sr örnek

        segment = wav[:samples_per_segment]
            
            # Eğer segment 5 saniyeden kısa ise, sıfırlarla doldur
        if len(segment) < samples_per_segment:
            segment = np.pad(segment, ( samples_per_segment - len(segment),0))
            
            
        return segment    


eda=EDA()
train,valid=train_test_split(data,test_size=0.15)


from torch.utils.data import Dataset, DataLoader
class CreatetDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = f"/kaggle/input/birdclef-2025/train_audio/{row['filename']}"

        if os.path.exists(audio_path):
            video_tensor,label = eda.sub_eda2(audio_path)
        return video_tensor, label

train_dataset = CreatetDataset(train)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataset = CreatetDataset(valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)


class AudioEfficientNet(nn.Module):
    def __init__(self, num_classes=206):
        super(AudioEfficientNet, self).__init__()
        
        # EfficientNet modelini yükle
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=True, in_chans=1)

        # Son katmanı sınıf sayısına göre değiştir
        self.efficientnet.classifier = nn.Linear(self.efficientnet.classifier.in_features, num_classes)

    def forward(self, x):
         # (Batch, Channel=1, Frequency=165, Time=2501)
        x = self.efficientnet(x)
        return x
#efficient=AudioEfficientNet()


efficient = torch.load("/kaggle/input/full_and_last/pytorch/default/1/full_model_fin.pth")
efficient.to(ConfigMein.device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(efficient.parameters(), lr=1e-4) 
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()
num_epochs = 5

def train_model(model, train_loader, valid_loader, epochs):
    
    model.to(ConfigMein.device)
    batch_size = 8
    m = nn.Softmax(dim=1)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, total=len(train_loader), 
                            desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for (videos, labels) in progress_bar:
            videos, labels = videos.to(ConfigMein.device), labels.to(ConfigMein.device) # BCE için float() gerekli olabilir

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(videos) # Çıkış sigmoid ile olasılık
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
                      
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            predicted = m(outputs).argmax(-1)  # Binary thresholding
            correct += (predicted == labels.argmax(-1)).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=f"{running_loss / (total / batch_size):.4f}", 
                                     acc=f"{correct/total:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

        # Validation
        model.eval()
        running_loss_v = 0.0
        correct_v = 0
        total_v = 0

        progress_bar_val = tqdm(valid_loader, total=len(valid_loader), 
                                desc=f"Validation {epoch+1}/{epochs}", leave=False)

        with torch.no_grad():
            for (videos, labels) in progress_bar_val:
                videos, labels = videos.to(ConfigMein.device), labels.to(ConfigMein.device)

                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                running_loss_v += loss.item()
                predicted = m(outputs).argmax(-1)
                correct_v += (predicted == labels.argmax(-1)).sum().item()
                total_v += labels.size(0)

                progress_bar_val.set_postfix(loss=f"{running_loss_v / (total_v / batch_size):.4f}", 
                                             acc=f"{correct_v/total_v:.4f}")

         # ReduceLROnPlateau, validation loss almalı

        if (epoch + 1) % 3 == 0:
            print("Model Checkpointed")
            torch.save(model.state_dict(), f"longshort_{epoch}.pth")

        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {running_loss_v/len(valid_loader):.4f}, Val Accuracy: {correct_v/total_v:.4f}")

    print("Training Complete!")



#train_model(efficient,train_dataloader,valid_dataloader,1)


import glob
if(len(glob.glob("/kaggle/input/birdclef-2025/test_soundscapes/*.ogg"))>0):
    data_path="/kaggle/input/birdclef-2025/test_soundscapes"
    mode="Testfor"
    sub_paths=glob.glob(f"{data_path}/*.ogg")
else:
    data_path="/kaggle/input/birdclef-2025/train_soundscapes"
    mode="Trainfor"
    sub_paths=glob.glob(f"{data_path}/*.ogg")[:5]


print(len(sub_paths))



class IterDataset(IterableDataset):
    def __init__(self,sub_paths,mode):
        self.paths=sub_paths
        self.mode=mode
    def __iter__(self):
        tensor=eda.for_test(self.paths,self.mode)
        return tensor
        
test_dataset = IterDataset(sub_paths,mode)




def test_model(model, dataset,mode, csv_path="submission.csv"):
    m = nn.Softmax(dim=1)
    model.to(ConfigMein.device)
    model.eval()
    
        
    liste_tensor=[]
    liste_label=[]
    for tensor, index in DataLoader(dataset, batch_size=8):
        tensor = tensor.to(ConfigMein.device)  # Eğer GPU kullanıyorsan
        result = m(model(tensor))
        result = result.detach().cpu().numpy()# Softmax uygula
            
        for idx in range(len(result)):
            liste_tensor.append(result[idx])
            liste_label.append(index[idx])
    return liste_tensor,liste_label  
   

liste_tensor,liste_label=test_model(efficient,test_dataset,mode,"submission.csv")


len(liste_tensor),len(liste_label)


len(liste_tensor[0])


if(liste_tensor is not None):
    submission = pd.DataFrame()
    submission["row_id"] = liste_label  # row_id ekle
    
    # liste_tensor'u DataFrame'e çevirerek sütunlarla uyumlu hale getir
    tensor_df = pd.DataFrame(liste_tensor, columns=prob_labels.columns[:len(liste_tensor[0])])
    
    # submission ile birleştir
    submission = pd.concat([submission, tensor_df], axis=1)


submission.to_csv("/kaggle/working/submission.csv",index=False)


submission.iloc[4][submission.columns[1:]].argmax()




