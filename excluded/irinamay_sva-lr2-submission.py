import numpy as np
import librosa as lb
import soundfile as sf
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from  torch.utils.data import Dataset
import torchvision.models as models
import os
from tqdm import tqdm


THRESH = 0.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)
test_audio_path = Path("../input/birdclef-2021/test_soundscapes")
sample_path = "../input/birdclef-2021/sample_submission.csv"
train_soundscapes_path = None
if not len(list(test_audio_path.glob("*.ogg"))):
    test_audio_path = Path("../input/birdclef-2021/train_soundscapes")
    sample_path = None
    train_soundscapes_path = Path("../input/birdclef-2021/train_soundscape_labels.csv")


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):

        melspec = lb.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length):
    if len(y) < length:
        y = np.concatenate([y, length - np.zeros(len(y))])
    elif len(y) > length:
        y = y[:length]
    return y


class BirdCLEFDataset(Dataset):
    def __init__(self, data, sr=32000, n_mels=128, fmin=0, fmax=None, duration=5, step=None, res_type="kaiser_fast", resample=True):
        
        self.data = data
        
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin,
                                                 fmax=self.fmax)
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image
    
    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio) 
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(y=audio, orig_sr=orig_sr, target_sr=self.sr, res_type=self.res_type)
          
        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])
            
        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]
            
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)
        
        return images
    
        
    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])


def load_net_universal(checkpoint_path, num_classes=397):
    net = models.efficientnet_b2(weights=None)
    net.classifier = nn.Linear(net.classifier[1].in_features, num_classes)
    
    dummy_device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=dummy_device)
    
    filename = Path(checkpoint_path).name
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loading new format: {filename}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}, Metric: {checkpoint.get('metric', 'N/A')}")
    else:
        state_dict = checkpoint
        print(f"Loading old format: {filename}")
    
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    net.load_state_dict(state_dict)
    net = net.to(device)
    net = net.eval()
    
    return net


@torch.no_grad()
def get_thresh_preds(out, thresh=None):
    thresh = thresh or THRESH
    o = (-out).argsort(1)
    npreds = (out > thresh).sum(1)
    preds = []
    for oo, npred in zip(o, npreds):
        preds.append(oo[:npred].cpu().numpy().tolist())
    return preds

def get_bird_names(preds):
    bird_names = []
    for pred in preds:
        if not pred:
            bird_names.append("nocall")
        else:
            bird_names.append(" ".join([inv_label_ids[bird_id] for bird_id in pred]))
    return bird_names

def predict(nets, test_data, names=True):
    preds = []
    with torch.no_grad():
        for idx in  tqdm(list(range(len(test_data)))):
            xb = torch.from_numpy(test_data[idx]).to(device)
            pred = 0.
            for net in nets:
                o = net(xb)
                o = torch.sigmoid(o)

                pred += o

            pred /= len(nets)
            
            if names:
                pred = get_bird_names(get_thresh_preds(pred))

            preds.append(pred)
    return preds


def preds_as_df(data, preds):
    sub = {
        "row_id": [],
        "birds": [],
    }
    
    for row, pred in zip(data.itertuples(False), preds):
        row_id = [f"{row.id}_{row.site}_{5*i}" for i in range(1, len(pred)+1)]
        sub["birds"] += pred
        sub["row_id"] += row_id
        
    sub = pd.DataFrame(sub)
    
    if sample_path:
        sample_sub = pd.read_csv(sample_path, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
        sub["birds"] = sub["birds"].fillna("nocall")
    return sub


data = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(test_audio_path).glob("*.ogg")],
    columns = ["filename", "id", "site", "date", "filepath"]
)
print(data.shape)


df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")

label_ids = {label: label_id for label_id,label in enumerate(sorted(df_train["primary_label"].unique()))}
inv_label_ids = {val: key for key,val in label_ids.items()}


test_data = BirdCLEFDataset(data=data)
len(test_data), test_data[0].shape


# ансамбль моделей, обученных на разных папках с данными
checkpoint_paths = [
    Path('/kaggle/input/bird-model-fold3-epoch23/pytorch/default/1/birdclef_efficientnet_fold3_epoch23_0.7138.pth'),
    Path('/kaggle/input/modelele/pytorch/default/1/birdclef_efficientnet_fold0_epoch_24_f1_val_07082_20251116210814.pth')
]


nets = [
        load_net_universal(checkpoint_path.as_posix()) for checkpoint_path in checkpoint_paths
]


pred_probas = predict(nets, test_data, names=False)
print(len(pred_probas))


preds = [get_bird_names(get_thresh_preds(pred, thresh=THRESH)) for pred in pred_probas]


# if train_soundscapes_path and Path(train_soundscapes_path).exists():
#     true_labels_df = pd.read_csv(train_soundscapes_path)
    
#     print("Columns in true_labels_df:", true_labels_df.columns.tolist())
#     print("Columns in data:", data.columns.tolist())
    
#     print("\n=== DEBUG INFO ===")
#     print("First 5 filenames in data:")
#     for i in range(min(5, len(data))):
#         print(f"  {data.loc[i, 'filename']}")
    
#     print("\nFirst 5 audio_id in true_labels_df:")
#     for audio_id in true_labels_df['audio_id'].unique()[:5]:
#         print(f"  {audio_id}")
    
#     correct = 0
#     total = 0
#     matched_files = 0
    
#     for idx, pred_birds_list in enumerate(preds):
#         filename = data.loc[idx, "filename"]
#         audio_id = int(filename.split('_')[0])
        
#         if idx < 3:
#             print(f"\nChecking file {idx}: {filename} -> audio_id: {audio_id}")
        
#         true_labels = true_labels_df[true_labels_df['audio_id'] == audio_id]
        
#         if len(true_labels) == 0:
#             if idx < 3:
#                 print(f"  No true labels found for audio_id: {audio_id}")
#             continue
#         else:
#             matched_files += 1
#             if idx < 3:
#                 print(f"  Found {len(true_labels)} true labels")
#                 print(f"  First few true labels:")
#                 print(true_labels.head(3)[['audio_id', 'seconds', 'birds']])
        
#         # для каждого 5-секундного сегмента
#         for seg_idx, pred_birds in enumerate(pred_birds_list):
#             seconds = seg_idx * 5
            
#             true_row = true_labels[true_labels['seconds'] == seconds]
            
#             if len(true_row) > 0:
#                 true_birds = true_row['birds'].values[0]
                
#                 pred_set = set(pred_birds.split()) if pred_birds != "nocall" else set()
#                 true_set = set(true_birds.split()) if true_birds != "nocall" else set()
                
#                 if idx < 3 and seg_idx < 3:  # DEBUG первые сегменты
#                     print(f"  Segment {seconds}s: Pred={pred_set}, True={true_set}, Match={pred_set == true_set}")
                
#                 if pred_set == true_set:
#                     correct += 1
#                 total += 1
    
#     print(f"\n=== FINAL RESULTS ===")
#     print(f"Files matched: {matched_files}/{len(data)}")
#     print(f"Segments matched: {correct}/{total}")
#     if total > 0:
#         print(f"Accuracy: {correct/total:.4f}")
#     else:
#         print("No segments matched - check audio_id conversion!")


sub = preds_as_df(data, preds)
print(sub.shape)
sub


sub.to_csv("submission.csv", index=False)

