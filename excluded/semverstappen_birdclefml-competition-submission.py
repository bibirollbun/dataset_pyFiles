import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd


#Parameters
TEST_DIR = '/kaggle/input/birdclef-2025/test_soundscapes/'
MODEL_PATH = '/kaggle/input/birdclef_stratifiedcnn/pytorch/stratified/2/birdclef_cnnstratified.pth'#The model with stratification will automatically be loaded
SUBMIT_PATH = '/kaggle/working/submission.csv'
SR = 32000 #Sampling rate 32K hz
CHUNK_LEN = 5 #5 seconds
N_MELS = 128 #the amount of mel frequency bands used to convert audios to spectrogram
BATCH_SIZE = 32 #Choose batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Define when to use gpu or cpu


#Load label mapping from training
unique_labels = [
    "1139490", "1192948", "1194042", "126247", "1346504", "134933", "135045", "1462711", "1462737", "1564122",
    "21038", "21116", "21211", "22333", "22973", "22976", "24272", "24292", "24322", "41663",
    "41778", "41970", "42007", "42087", "42113", "46010", "47067", "476537", "476538", "48124",
    "50186", "517119", "523060", "528041", "52884", "548639", "555086", "555142", "566513", "64862",
    "65336", "65344", "65349", "65373", "65419", "65448", "65547", "65962", "66016", "66531",
    "66578", "66893", "67082", "67252", "714022", "715170", "787625", "81930", "868458", "963335",
    "amakin1", "amekes", "ampkin1", "anhing", "babwar", "bafibi1", "banana", "baymac", "bbwduc", "bicwre1",
    "bkcdon", "bkmtou1", "blbgra1", "blbwre1", "blcant4", "blchaw1", "blcjay1", "blctit1", "blhpar1", "blkvul",
    "bobfly1", "bobher1", "brtpar1", "bubcur1", "bubwre1", "bucmot3", "bugtan", "butsal1", "cargra1", "cattyr",
    "chbant1", "chfmac1", "cinbec1", "cocher1", "cocwoo1", "colara1", "colcha1", "compau", "compot1", "cotfly1",
    "crbtan1", "crcwoo1", "crebob1", "cregua1", "creoro1", "eardov1", "fotfly", "gohman1", "grasal4", "grbhaw1",
    "greani1", "greegr", "greibi1", "grekis", "grepot1", "gretin1", "grnkin", "grysee1", "gybmar", "gycwor1",
    "labter1", "laufal1", "leagre", "linwoo1", "littin1", "mastit1", "neocor", "norscr1", "olipic1", "orcpar",
    "palhor2", "paltan1", "pavpig2", "piepuf1", "pirfly1", "piwtyr1", "plbwoo1", "plctan1", "plukit1", "purgal2",
    "ragmac1", "rebbla1", "recwoo1", "rinkin1", "roahaw", "rosspo1", "royfly1", "rtlhum", "rubsee1", "rufmot1",
    "rugdov", "rumfly1", "ruther1", "rutjac1", "rutpuf1", "saffin", "sahpar1", "savhaw1", "secfly1", "shghum1",
    "shtfly1", "smbani", "snoegr", "sobtyr1", "socfly1", "solsan", "soulap1", "spbwoo1", "speowl1", "spepar1",
    "srwswa1", "stbwoo2", "strcuc1", "strfly1", "strher", "strowl1", "tbsfin1", "thbeup1", "thlsch3", "trokin",
    "tropar", "trsowl", "turvul", "verfly", "watjac1", "wbwwre1", "whbant1", "whbman1", "whfant1", "whmtyr1",
    "whtdov", "whttro1", "whwswa1", "woosto", "y00678", "yebela1", "yebfly1", "yebsee1", "yecspi2", "yectyr1",
    "yehbla2", "yehcar1", "yelori1", "yeofly1", "yercac1", "ywcpar"
]

idx2label = {i: label for i, label in enumerate(unique_labels)}


#Helper functions
def split_audio(audio, sr=SR, chunk_length=CHUNK_LEN):
    samples_per_chunk = chunk_length * sr
    return [audio[i:i+samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]

def to_mel_spectrogram(chunk):
    mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db -= mel_db.min()
    mel_db /= mel_db.max()
    return mel_db

#Defining model architecture
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#Loading model obtained from training script
model = CNNModel(num_classes=len(unique_labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


#Iterates over test files, filtering for .ogg
rows = []
for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.endswith('.ogg'):
        continue
        
    #Loads audio files and splits into chunks
    path = os.path.join(TEST_DIR, fname)
    audio, _ = librosa.load(path, sr=SR)
    chunks = split_audio(audio)
    
    #Extracting the soundscape ID's
    soundscape_id = fname.split('_')[1].split('.')[0]

    #Pad audio if audio is too short
    for i, chunk in enumerate(chunks):
        if len(chunk) < CHUNK_LEN * SR:
            chunk = np.pad(chunk, (0, CHUNK_LEN * SR - len(chunk)))

        #Convert to mel_spectrogram
        mel = to_mel_spectrogram(chunk)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

        #Run the model prediction
        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        #Row_id formatting: soundscape_[soundscape_id]_[end_time]
        end_time = (i + 1) * CHUNK_LEN
        row_id = f"soundscape_{soundscape_id}_{end_time}"

        row = [row_id] + probs.tolist()
        rows.append(row)

#Create the submission dataframe and save this 
columns = ['row_id'] + unique_labels
df = pd.DataFrame(rows, columns=columns)
df.to_csv(SUBMIT_PATH, index=False)
print(f"✅ Submission saved to {SUBMIT_PATH}")

