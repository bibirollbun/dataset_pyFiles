# %config Completer.use_jedi = False


# # Reading in meta-data file

# import numpy as np
# import pandas as pd
# meta_data = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")


# meta_data.head()


# meta_data.shape


# import os
# import librosa
# import numpy as np
# import pandas as pd


# # Load in the soundscape data with librosa understand the data we're dealing with
# # What does librosa.load really return: 
# # https://stackoverflow.com/questions/61986490/what-does-librosa-load-return
# np.random.seed(42)

# signal, rate = librosa.load(path="/kaggle/input/birdclef-2025/train_audio/1139490/CSA36385.ogg", sr=None)


# from IPython.display import Audio

# Audio(data=signal, rate=rate)


# import matplotlib.pyplot as plt
# sampling_rate = rate
# time_step = 1 / sampling_rate

# #Generate time vector
# time = np.linspace(0, len(signal)/rate, num=len(signal))

# plt.figure(figsize=(10, 4))
# plt.plot(time, signal)
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")
# plt.title("Audio wavefrom after transformation")
# plt.show()


# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(signal, sr=sampling_rate, axis='time')
# plt.title('Audio Waveform')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.show()


# FRAME_SIZE = 2048
# HOP_SIZE = 512
# stft_data = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
# stft_data.shape


# stft_data_ammplitude = np.abs(stft_data) ** 2


# # Visualizating the spectrogram
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(stft_data_ammplitude, 
#                          sr=sampling_rate,
#                          hop_length=HOP_SIZE,
#                          x_axis="time", 
#                          y_axis="linear")
# plt.colorbar(format="%+2.f")


# # Let's do it in log scale
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(librosa.power_to_db(stft_data_ammplitude), 
#                          sr=sampling_rate,
#                          hop_length=HOP_SIZE,
#                          x_axis="time", 
#                          y_axis="linear")
# plt.colorbar(format="%+2.f")


# plt.figure(figsize=(25, 10))
# librosa.display.specshow(librosa.power_to_db(stft_data_ammplitude), 
#                          sr=sampling_rate,
#                          hop_length=HOP_SIZE,
#                          x_axis="time", 
#                          y_axis="log")
# plt.colorbar(format="%+2.f")


# mffc_features = librosa.feature.mfcc(y=signal, sr=sampling_rate)


# import os
# with open("/kaggle/working/audio_file_name.txt", "w") as a_file:
#     for path, subdirs, files in os.walk('/kaggle/input/birdclef-2025/train_audio/'):
#         for filename in files:
#             f = os.path.join(path, filename)
#             a_file.write(str(f) + os.linesep)
            


import os
dirs = []
for directory in os.listdir('/kaggle/input/birdclef-2025/train_audio/'):
    dirs.append(directory)
dirs.sort()
print(dirs)


import time
import numpy as np
import pandas as pd
import librosa
import math, random
import torch
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt
import ast
from IPython.display import Audio, display
from torch.utils.data import DataLoader, Dataset, random_split

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


# # Input for the filepath
# train_folder = "/kaggle/input/birdclef-2025/train_audio"
# label = "creoro1"
# i = 0
# for file in sorted(os.listdir(os.path.join(train_folder, label))):
#     signal, rate = librosa.load(os.path.join(train_folder, label, file), sr=None)
#     print(f" The is file {file} with sampling rate of {rate}")
#     display(Audio(data=signal, rate=rate))
#     i += 1
#     if i == 15:
#         break


train_label_1 = pd.read_csv("/kaggle/input/birdclef-label/train_label_1.csv")
train_label_2 = pd.read_csv("/kaggle/input/birdclef-label/train_label_2.csv")


train_label_1.dropna(subset=["offset_time"], inplace=True)
train_label_2.dropna(subset=["offset_time"], inplace=True)


# print(len(set(train_label_1["primary_label"])))
# print(len(set(train_label_2["primary_label"])))
print(sorted(list(set(train_label_1["primary_label"]))))
print(sorted(list(set(train_label_2["primary_label"]))))


train_label = pd.concat([train_label_1, train_label_2], axis=0)
train_label.shape


# Todo1: Build the meta data csv file
columns_name = ["FileName", "Offset", "Labels"]
train_df = pd.DataFrame(columns=columns_name)

for index, row in train_label.iterrows():
    offset_time = str(row["offset_time"]).split(",")
    labels = tuple([str(row['primary_label'])] + ast.literal_eval(row['secondary_labels']))
    
    for offset in offset_time:
        new_row = pd.DataFrame([[row['filename'], offset, labels]], columns=columns_name)
        train_df = pd.concat([train_df, new_row], axis=0)



# # Inivestigate the issues with how I lable the data and fix it
# list(train_df["Offset"].unique())


# Couple of issues:
# 1) drop the missing offset data
# 2) one instance where the offset time is separate by dot instead of comma, separate that
# 3) convert everything in offset column from string type to integer type
# 4) Reset the index, very important later when the Deep Learning model do the fit

# 1
train_df = train_df[train_df['Offset'] != ' ']

# 2
weird_instance = train_df[train_df["Offset"] == "5. 25"].reset_index(drop=True)
for offset in [5, 25]:
    new_row = pd.DataFrame([[weird_instance.loc[0,"FileName"], offset, weird_instance.loc[0,"Labels"]]], columns=columns_name)
    train_df = pd.concat([train_df, new_row], axis=0)
train_df = train_df[train_df['Offset'] != '5. 25']

# 3
train_df["Offset"] = train_df["Offset"].astype(int)

# 4
train_df.reset_index(drop=True, inplace=True)


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=dirs)
mlb.fit(train_df["Labels"])
train_df_labels = mlb.transform(train_df["Labels"])


# Todo2: Transform data before feeding into the model. Those includes
# 1) Load the files
# 2) Resample and convert to stereo. I know all the data I'm fitting in is 32kHz sampling rate, 
#    but I need to double check whether it's mono (1 channel) or stereo (2 channel). You can
#    know it by the shape of the signal
# 3) Resize to fixed length. I already know that I only want a couple of 5 second long portion of each file
# 4) Data Augmentation to create more data. Do I need this
# 5) Feature Extraction: Ok now the data is clean and the latest techniques is to convert it to mel spectrogram
# 6) Another round of data augmentation with masking
class AudioUtil():
    @staticmethod
    def load_audio(path, offset=None, duration=None):
        # Basically load with native sampling rate
        signal, sampling_rate = librosa.load(path, offset=offset, duration=duration, sr=None)
        return (torch.from_numpy(signal.reshape(1, -1)), sampling_rate)

    @staticmethod
    def rechannel(audio, new_channel):
        '''
        audio: 
            - signal (either mono or stereo)
            - sampling_rate - in Hrtz
        new_channel: number of new channel we want to conver to
        '''
        sig, sr = audio
        if sig.shape[0] == new_channel:
            return audio


        if new_channel == 1:
            # Convert to mono if only usinng the first channel
            resig = sig[:1, :]
        else:
            # Duplicate the first channel to make it stereo
            resig = torch.cat((sig, sig))
        return resig

    @staticmethod
    def resample(audio, newsr):
        '''
        audio: 
            - signal (either mono or stereo)
            - sampling_rate - in Hrtz
        newsr: number of new channel we want to convert to
        '''
        sig, sr = audio
        if sr == newsr:
            return audio

        num_channels = sig.shape[0]
        resig = torchaudio.transform.Resample(sr, newsr)(sig[:1, :])
        if(num_channels > 1):
            retwo = torchaudio.transform.Resample(sr, newsr)(sig[:1, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len >= max_len):
            # truncat
            sig = sig[:,:max_len]
        else:
            # pad
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
    
            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=2048, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spectrogram shape [channel, n_mels, time] where channel is mono, stereo
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_mask=1, n_time_mask=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
            
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
          aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
          aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    
        return aug_spec





# Todo3: Let's build the Dataset. One of the building block for Pytorch 
# We want this because we can't just load everything in RAM. For deep learning, just load it while training
class SoundDS(Dataset):
    def __init__(self, df, df_labels, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 5000 # 5 second long
        self.sr = 32000 # sampling rate provided in 32 kHz
        self.channel = 1
        self.shift_pct = 0.4
        self.df_labels = df_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Purpose of this function is to get the data of the file and corresponding ID
        '''
        audio_file = os.path.join(self.data_path, self.df.loc[idx, 'FileName'])
        class_id = torch.tensor(self.df_labels[idx, :], dtype=torch.float)

        aud = AudioUtil.load_audio(audio_file, offset=self.df.loc[idx, 'Offset'], duration=5)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan= AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=128, n_fft=1024, hop_len=None)
        
        return sgram, class_id
      


# Todo 4: Prepare batches of data
train_folder = "/kaggle/input/birdclef-2025/train_audio"
myds = SoundDS(train_df, train_df_labels, train_folder)

# Random split of 80:20
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val], generator=torch.Generator().manual_seed(42))

# Training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)


# Todo 5: Create Model, CNN with 4 convoluted layer
class AudioClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers= []

        # First convolution block with Relu and Batch Norm. Use Kaiming initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=206)
        self.sigmoid = nn.Sigmoid()

        # Wrap everything together
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # run convolution blocks
        x = self.conv(x)

        # adaptive pooling and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear Layer and then sigomoid?
        x = self.lin(x)
        x = self.sigmoid(x)
        return x

myModel = AudioClassifer()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

# Check that it is on Cuda
next(myModel.parameters()).device


# Todo 6: Define the evaluation metrics. Macro-avearged ROC-AUC score
from sklearn.metrics import roc_auc_score
def macro_roc_auc_score(y_true, y_pred_proba):
    n_classes = y_true.shape[1]
    auc_scores = []
    for i in range(n_classes):
        if np.unique(y_true[:,i]).size > 1:
            auc = roc_auc_score(y_true[:,i], y_pred_proba[:,i])
            auc_scores.append(auc)
    return np.mean(auc_scores)



# Todo 6: Training
def training(model, train_dl, num_epochs):
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=0.007)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=0.007,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='cos')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0

        # Repeat for each batch in training set
        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize data
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
        
            # Zero parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Keep stats for Loss and Accuracy
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_dl)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.5f}')
    print("Finished Training")

start_time = time.time()
num_epochs = 110
training(myModel, train_dl, num_epochs)
end_time = time.time()
print(f"Elapsed time for training: {end_time - start_time} seconds")


# # Todo 7: Inference for one instance. Helpful for submitting the model
# val_features, val_labels = next(iter(val_dl))
# with torch.no_grad():
#     inputs, labels = val_features[0].to(device), val_labels[0].to(device)
#     inputs = inputs.unsqueeze(0)
    
#     # Normalize the inputs
#     inputs_m, inputs_s = inputs.mean(), inputs.std()
#     inputs = (inputs - inputs_m) / inputs_s

#     outputs = myModel(inputs)
#     print(f"Shape of outputs: {outputs.shape}")
#     print(f"Shape of labels: {labels.shape}")
    
#     # print(torch.flatten(outputs).tolist())
    
    


# Todo 7: Inference for the whole val_ds DataLoader instance. Helpful for submitting the model
from torchmetrics.classification import MultilabelAUROC
criterion = nn.BCELoss()
val_loss = 0.0
all_outputs = []
all_labels = []
with torch.no_grad():
    for data in val_dl:
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
    
        outputs = myModel(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        # print(f"Shape of outputs: {outputs.shape}")
        # print(f"Shape of labels: {labels.shape}")
        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())
all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

num_labels = all_outputs.shape[1]
auroc_metric = MultilabelAUROC(num_labels=num_labels, average='macro')

macro_auc = auroc_metric(all_outputs, all_labels.int()).item()

print(f"Avearge macro scores among all validation batches is: {macro_auc}")
print(f"Validation Loss is: {val_loss/len(val_dl)}")


# Todo: Integrate validation loss



# # Save the model
# model_path = "/kaggle/working/audio_classifier.pth"
# torch.save(myModel.state_dict(), model_path)


# # Load the model
# model_path = "/kaggle/input/bird_classification/pytorch/default/1/audio_classifier.pth"
# myModel = AudioClassifer()
# myModel.load_state_dict(torch.load(model_path, weights_only=True))
# myModel.eval()


# Todo: Submit the prediction based upon the sample of the competition
# Set seed
np.random.seed(42)

# Class labels from train audio
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

# List of test soundscapes (only visible during submission)
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=['row_id'] + class_labels)
for soundscape in test_soundscapes:

    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(chunk)
        
    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):
        
        # Get row id  (soundscape id + end time of 5s chunk)      
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
        
        # Make prediction (let's use random scores for now)
        # Preprocess the data > Feed the data into model > append prediction
        chunk = torch.from_numpy(chunk.reshape(1, -1))
        aud = (chunk, rate)
        reaud = AudioUtil.resample(aud, rate)
        rechan= AudioUtil.rechannel(reaud, 1)
        dur_aud = AudioUtil.pad_trunc(rechan, 5000)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

        with torch.no_grad():
            inputs = sgram.unsqueeze(0)
            
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
        
            outputs = myModel(inputs)

        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + torch.flatten(outputs).tolist()], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
        
# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
predictions.head()




