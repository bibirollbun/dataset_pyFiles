import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random
import timm

torch.manual_seed(42);


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE


df = pd.read_parquet('/kaggle/input/bc25-eda/train_metadata_joined.parquet')

SPEC_CROP = 248

IDX_TO_LABEL = sorted(df.primary_label.unique())
LABEL_TO_IDX = dict((v, k) for k, v in enumerate(IDX_TO_LABEL))
NUM_LABELS = len(IDX_TO_LABEL)
LABEL_TO_ONEHOT = torch.eye(NUM_LABELS, device=DEVICE)
LABELS = torch.tensor([LABEL_TO_IDX[l] for l in df.primary_label.iloc], device=DEVICE)
MIN_SPEC_VALUES = torch.tensor(df.min_spec_value, dtype=torch.float32, device=DEVICE)
MAX_SPEC_VALUES = torch.tensor(df.max_spec_value, dtype=torch.float32, device=DEVICE)

NUM_SAMPLES = len(df)
split = int(0.8 * NUM_SAMPLES)
all_indices = torch.randperm(NUM_SAMPLES)
TRAIN_INDICES = all_indices[:split]
VAL_INDICES = all_indices[split:]


def load_file(filename):
    filename = '/kaggle/input/bc25-train-specs-signed-db-80-n-16/train_audio_specs/' + filename.split('.')[0] + '.pt'
    return torch.load(filename, weights_only=True, map_location=DEVICE)

ALL_SPECS = Parallel(n_jobs=-1)(
    delayed(load_file)(fname) for fname in tqdm(df.filename, desc="Loading files")
)


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


class SpecDataset(torch.utils.data.Dataset):
    def __init__(self, validation):
        self.validation = validation
    def __getitem__(self, i):
        spec = ALL_SPECS[i]
        max_range = spec.shape[1] - SPEC_CROP
        if self.validation:
            spec = spec[:, :SPEC_CROP] # only first 5 sec
        else:
            start = random.randint(0, max(0, max_range))
            spec = spec[:, start:start+SPEC_CROP] # random crop
        if max_range < 0:
            spec = torch.nn.functional.pad(spec, (0, -max_range)) # fill with zeros
        spec = q.dequantize(spec, MIN_SPEC_VALUES[i], MAX_SPEC_VALUES[i])
        return spec[None], LABEL_TO_ONEHOT[LABELS[i]]
    def __len__(self):
        return len(df)


def make_dataloaders():
    train_dataset = torch.utils.data.Subset(SpecDataset(validation=False), TRAIN_INDICES)
    val_dataset = torch.utils.data.Subset(SpecDataset(validation=True), VAL_INDICES)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    return train_dataloader, val_dataloader


batch = next(iter(make_dataloaders()[1]))
plt.imshow(batch[0][0][0].to('cpu'))
plt.colorbar();


def make_model():
    return timm.create_model(
        'tf_efficientnet_b0',
        in_chans=1,
        num_classes=NUM_LABELS,
        pretrained=False,
    )

make_model()(batch[0].to('cpu')).shape


def get_val_metrics(logits, labels, lossfn):
    loss = lossfn(logits, labels).item()
    acc = (labels == (logits > 0.0)).float().mean().item()
    return dict(loss=loss, acc=acc)


def train(model, optimizer, num_epochs, train_losses=[], val_metrics=[]):
    train_dataloader, val_dataloader = make_dataloaders()
    lossfn = torch.nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for specs, labels in train_dataloader:
            optimizer.zero_grad()
            logits = model(specs)
            loss = lossfn(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().item())
        model.eval()
        with torch.inference_mode():
            epoch_labels = []
            epoch_logits = []
            for specs, labels in val_dataloader:
                logits = model(specs)
                epoch_labels.append(labels)
                epoch_logits.append(logits)
            epoch_labels = torch.concat(epoch_labels)
            epoch_logits = torch.concat(epoch_logits)
            val_metrics.append(get_val_metrics(epoch_logits, epoch_labels, lossfn))
        torch.save(model.state_dict(), f'model_state_dict_epoch_{epoch}.pt')
    return train_losses, val_metrics

def plot_losses(train_losses, val_metrics):
    num_train_batches = len(make_dataloaders()[0])
    plt.plot(train_losses)
    plt.plot([v['loss'] for v in val_metrics for _ in range(num_train_batches)])



model = make_model().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_losses, val_metrics = train(model, optimizer, 20)
plot_losses(train_losses, val_metrics)




