IS_SUBMISSION = False
FOLD = 0 # -1 means take all folds and ensemble them
#NUM_TEST_FILES = 700


import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import timm
import os
from glob import glob
import json
from types import SimpleNamespace
import torchvision
import itertools

# Set seed
np.random.seed(42)
torch.manual_seed(42);


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE


IDX_TO_LABEL = sorted(pd.read_csv('/kaggle/input/birdclef-2025/train.csv').primary_label.unique())
NUM_LABELS = len(IDX_TO_LABEL)


global_model_dir = glob('/kaggle/input/bc25-train-*')[0]
global_model_dir


with open(f'{global_model_dir}/CFG.json', 'r') as file:
    CFG = json.load(file)
CFG = SimpleNamespace(**CFG)
CFG


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            in_chans=1,
            drop_rate=0.2,
            drop_path_rate=0.2
        )
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Identity()
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = torch.nn.Linear(backbone_out, NUM_LABELS)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def make_model():
    if CFG.MODEL_VERSION == 0:
        return timm.create_model(
            'tf_efficientnet_b0',
            in_chans=1,
            num_classes=NUM_LABELS,
            pretrained=False,
        )
    elif CFG.MODEL_VERSION == 1:
        return Model1()
    elif CFG.MODEL_VERSION == 2:
        return timm.create_model(
            'tf_efficientnet_b0',
            in_chans=1,
            num_classes=NUM_LABELS,
            pretrained=False,
        )
    elif CFG.MODEL_VERSION == 3:
        return timm.create_model(
            'efficientnet_b0',
            in_chans=1,
            num_classes=NUM_LABELS,
            pretrained=False,
        )
    elif CFG.MODEL_VERSION == 6:
        return timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            in_chans=CFG.IN_CHANNELS,
            drop_rate=0.2,
            drop_path_rate=0.2,
            num_classes=NUM_LABELS,
        )
    elif CFG.MODEL_VERSION == 7:
        return timm.create_model(
            'efficientnet_b2',
            pretrained=False,
            in_chans=CFG.IN_CHANNELS,
            drop_rate=0.2,
            drop_path_rate=0.2,
            num_classes=NUM_LABELS,
        )


def load_model(fold):
    weight_paths = glob(f'/kaggle/input/bc25-train-*/model_state_dict_fold_{fold}_epoch_*.pt')
    model_dir = os.path.dirname(weight_paths[0])
    
    if hasattr(CFG, 'FULL_DATA') and CFG.FULL_DATA:
        best_epoch = max(
            int(e.split('_')[-1].split('.')[0])
            for e in glob(f'{model_dir}/model_state_dict_fold_{fold}_epoch_*.pt')
        )
    else:
        val_metrics = torch.load(f'{model_dir}/val_metrics_fold_{fold}.pt', weights_only=True)
        best_epoch = np.argmin([m['loss'] for m in val_metrics])

    weights_path = f'{model_dir}/model_state_dict_fold_{fold}_epoch_{best_epoch}.pt'
    print(weights_path)
    weights = torch.load(weights_path, weights_only=True, map_location=DEVICE)
    model = make_model().to(DEVICE)
    model.load_state_dict(weights)
    model.eval()
    return model


if FOLD == -1:
    models = Parallel(n_jobs=-1)(
        delayed(load_model)(fold) for fold in tqdm(range(5), desc="Loading models")
    )
else:
    models = [load_model(FOLD)]


to_spec = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=32000,
        n_mels=CFG.NUM_MELS,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        center=False,
        power=2,
    ),
    torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=80.0,
    )
).to(DEVICE)

resize_spec = (
    torchvision.transforms.Resize(CFG.RESIZE_TARGET) 
    if hasattr(CFG, 'RESIZE_TARGET') and CFG.RESIZE_TARGET 
    else None
)


"""
test_soundscape_path = (
    '/kaggle/input/birdclef-2025/test_soundscapes/' 
    if IS_SUBMISSION else 
    '/kaggle/input/birdclef-2025/train_soundscapes'
)
test_soundscapes = [
    os.path.join(test_soundscape_path, afile) 
    for afile in sorted(os.listdir(test_soundscape_path)) 
    if afile.endswith('.ogg')
]
test_soundscapes = (
    test_soundscapes
    if IS_SUBMISSION else
    test_soundscapes#[:NUM_TEST_FILES]
)
""";


df = pd.read_parquet('/kaggle/input/bc25-eda/train_metadata_joined.parquet')
test_soundscapes = [f'/kaggle/input/birdclef-2025/train_audio/{p}' for p in list(df.filename)]
test_soundscapes = test_soundscapes


class Quantizer:
    def __init__(self, num_bits):
        self.range = 2**num_bits
        self.max = 2**(num_bits - 1) - 1
        self.min = -2**(num_bits - 1)
        if num_bits <= 8:
            self.dtype = torch.int8
        elif num_bits <= 16:
            self.dtype = torch.int16
        elif num_bits <= 32:
            self.dtype = torch.int32

    def quantize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val: # Edge case: all values are the same
            return torch.full_like(tensor, 0, dtype=self.dtype), min_val, max_val
        scale = self.range / (max_val - min_val)
        quantized_tensor = torch.round((tensor - min_val) * scale + self.min).clamp(self.min, self.max).to(self.dtype)
        return quantized_tensor, min_val, max_val

    def dequantize(self, quantized_tensor, min_val, max_val):
        if min_val == max_val:
            return torch.full_like(quantized_tensor, min_val, dtype=torch.float32)
        scale = (max_val - min_val) / self.range
        return (quantized_tensor.to(torch.float32) - self.min) * scale + min_val


q = Quantizer(16)


import math

def circular_pad(x, n):
    s = x.shape[-1]
    n_extra = math.ceil(n / s) + 1
    y = torch.concat([x]*n_extra, axis=-1)
    return y[:s+n]

circular_pad(torch.arange(10), 5)


NUM_FRAMES = 32000*5

def get_spec_chunks(soundscape):
    audio = torchaudio.load(soundscape, num_frames=NUM_FRAMES)[0][0]
    max_range = audio.shape[0] - NUM_FRAMES
    if max_range < 0:
        audio = circular_pad(audio, -max_range)
    audio = audio.to(DEVICE).reshape(-1, 1, NUM_FRAMES)
    spec = to_spec(audio)
    return q.quantize(spec)

all_chunks = Parallel(n_jobs=-1)(
    delayed(get_spec_chunks)(f) for f in tqdm(test_soundscapes, desc="Loading files")
)
#all_chunks = list(itertools.chain.from_iterable(all_chunks)) # flatten
len(all_chunks), all_chunks[0][0].shape if len(all_chunks) > 0 else 0


FILE_IN_BATCH = 16


def prepare(chunk):
    spec = q.dequantize(*chunk)
    if CFG.IN_CHANNELS > 1:
        spec = spec.expand(-1, CFG.IN_CHANNELS, -1, -1)
    else:
        spec = spec[None]
    return spec[0]

probs = []
with torch.inference_mode():
    for i in tqdm(range(0, len(all_chunks), FILE_IN_BATCH), desc="Running inference"):
        batch = torch.stack([prepare(c) for c in all_chunks[i:i+FILE_IN_BATCH]])
        logits = torch.stack([model(batch) for model in models])
        #logits = stacked_model(batch)
        probs.append(torch.nn.functional.softmax(logits, dim=-1).mean(0))
        
    if len(probs) > 0:
        #probs = smoothen(torch.concat(probs)).to('cpu')
        probs = torch.concat(probs).to('cpu')


torch.save(probs, "probs.pt")


row_ids = []
for soundscape in test_soundscapes:
    for i in range(12):
        row_ids.append(os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}')

predictions = pd.DataFrame(probs, columns=IDX_TO_LABEL)
predictions['row_id'] = df.filename
predictions.to_csv('submission.csv', index=False)
predictions.head()


LABEL_TO_IDX = dict((v, k) for k, v in enumerate(IDX_TO_LABEL))


N = probs.shape[0]

top = probs.max(1)
top_predicted_idx = top[1]
top_predicted_label = [IDX_TO_LABEL[i] for i in top_predicted_idx]
top_predicted_prob = top[0]

primary_label = list(df.primary_label)[:N]
primary_label_prob = [probs[i, LABEL_TO_IDX[label]].item() for i, label in enumerate(primary_label)]

gt_idx = torch.tensor([LABEL_TO_IDX[l] for l in primary_label])
sorted_idx = probs.argsort(dim=1, descending=True)
primary_label_rank = (sorted_idx == gt_idx.unsqueeze(1)).nonzero()[:,1]

accuracy = [float(top_predicted_label[i] == primary_label[i]) for i in range(N)]
split = ['val' if f else 'train' for f in df.fold[:N] == FOLD]


metrics = pd.DataFrame(dict(
    filename=df.filename[:N],
    top_predicted_label=top_predicted_label,
    top_predicted_prob=top_predicted_prob,
    primary_label=primary_label,
    primary_label_prob=primary_label_prob,
    primary_label_rank=primary_label_rank,
    accuracy=accuracy,
    split=split
))
metrics.to_csv('metrics.csv', index=False)
metrics


metrics.groupby('split').accuracy.describe()




