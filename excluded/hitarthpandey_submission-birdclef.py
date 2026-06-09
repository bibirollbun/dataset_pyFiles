import os
import torch
import torchaudio
import pandas as pd
import torchaudio.transforms as T
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models

# --- Class Labels ---
class_names = [  # Full 206-class list
    "1139490", "1192948", "1194042", "126247", "1346504", "134933", "135045", "1462711", "1462737", "1564122",
    "21038", "21116", "21211", "22333", "22973", "22976", "24272", "24292", "24322", "41663", "41778", "41970", "42007",
    "42087", "42113", "46010", "47067", "476537", "476538", "48124", "50186", "517119", "523060", "528041", "52884",
    "548639", "555086", "555142", "566513", "64862", "65336", "65344", "65349", "65373", "65419", "65448", "65547",
    "65962", "66016", "66531", "66578", "66893", "67082", "67252", "714022", "715170", "787625", "81930", "868458",
    "963335", "amakin1", "amekes", "ampkin1", "anhing", "babwar", "bafibi1", "banana", "baymac", "bbwduc", "bicwre1",
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
    "yehbla2", "yehcar1", "yelori1", "yeofly1", "yercac1", "ywcpar"]

# --- Metadata Tensor ---
meta_cols = ['latitude', 'longitude', 'rating']
meta_df = pd.read_csv("/kaggle/input/fullmeta/subset_df_full_meta.csv")
meta_tensor = torch.tensor(meta_df[meta_cols].mean().values, dtype=torch.float32).unsqueeze(0)

# --- Model Definition (EffNetB3 + Metadata) ---
import torch.nn as nn
from timm import create_model

class EffB3ResNetEnsemble(nn.Module):
    def __init__(self, num_classes, metadata_dim):
        super().__init__()

        # EfficientNet-B3 backbone
        self.effb3 = create_model("efficientnet_b3", pretrained=False, in_chans=1, num_classes=0)

        # ResNet-18 backbone with single-channel input and no classification head
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Identity()

        # Metadata branch
        self.metadata_head = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        # Classifier head with smaller hidden layer and stronger dropout
        self.classifier = nn.Sequential(
            nn.Linear(1536 + 512 + 128, 256),   # ðŸ”½ smaller head for regularization
            nn.ReLU(),
            nn.Dropout(0.5),                    # ðŸ”¼ stronger dropout
            nn.Linear(256, num_classes)        # BCEWithLogits â†’ no sigmoid here
        )

    def forward(self, x, meta):
        feat_effb3 = self.effb3(x)
        feat_resnet = self.resnet18(x)
        feat_meta = self.metadata_head(meta)
        combined = torch.cat([feat_effb3, feat_resnet, feat_meta], dim=1)
        return self.classifier(combined)

# --- Load Model (CPU only) ---
device = torch.device("cpu")
model = EffB3ResNetEnsemble(num_classes=206, metadata_dim=3)
model.load_state_dict(torch.load("/kaggle/input/effb3resnet18/pytorch/default/1/best_model_ensemble_effb3_resnet.pth", map_location=device))
model.eval()

# --- Mel Spectrogram Settings ---
mel_transform = T.MelSpectrogram(
    sample_rate=32000,
    n_fft=1024,
    hop_length=320,
    n_mels=128,
    f_min=20,
    f_max=16000
)
db_transform = T.AmplitudeToDB()

# --- Load Test Soundscapes ---
test_path = "/kaggle/input/birdclef-2025/test_soundscapes"
soundscape_files = sorted([f for f in os.listdir(test_path) if f.endswith('.ogg')])
if not soundscape_files:
    test_path = "/kaggle/input/birdclef-2025/train_soundscapes"
    soundscape_files = sorted([f for f in os.listdir(test_path) if f.endswith('.ogg')])[:3]

# --- Inference ---
submission_rows = []

for fname in tqdm(soundscape_files):
    waveform, sr = torchaudio.load(os.path.join(test_path, fname))
    assert sr == 32000

    for start_sec in range(0, 60, 5):
        start_sample = start_sec * sr
        end_sample = (start_sec + 5) * sr
        if end_sample > waveform.shape[1]:
            continue

        clip = waveform[:, start_sample:end_sample]
        mel = mel_transform(clip)
        mel_db = db_transform(mel)
        x_img = mel_db.unsqueeze(0)  # [1, 1, 128, T]
        x_meta = meta_tensor

        with torch.no_grad():
            logits = model(x_img, x_meta)
            probs = torch.sigmoid(logits)[0].numpy()

        row_id = f"{fname.replace('.ogg', '')}_{start_sec + 5}"
        row = {"row_id": row_id}
        row.update({class_names[i]: probs[i] for i in range(206)})
        submission_rows.append(row)

# --- Save Submission ---
submission_df = pd.DataFrame(submission_rows)
submission_df = submission_df[["row_id"] + class_names]
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved with", len(submission_df), "rows")




