import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader


all_inputs = [
    f
    for f in
    Path('/kaggle/input/waveform-inversion/train_samples').rglob('*.npy')
    if ('seis' in f.stem) or ('data' in f.stem)
]


def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

all_outputs = inputs_files_to_output_files(all_inputs)


assert all(f.exists() for f in all_outputs)


train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)] # Sample every two
valid_inputs = [f for f in all_inputs if not f in train_inputs]


train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)


class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        # Calculate file offset and sample offset within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return X[sample_idx].copy(), y[sample_idx].copy()
        finally:
            del X, y


dstrain = SeismicDataset(train_inputs, train_outputs)
dltrain = DataLoader(dstrain, batch_size=64, shuffle=True, pin_memory=True, drop_last=True, num_workers=4, persistent_workers=True)

dsvalid = SeismicDataset(valid_inputs, valid_outputs)
dlvalid = DataLoader(dsvalid, batch_size=64, shuffle=False, pin_memory=True, drop_last=False, num_workers=4, persistent_workers=True)


# Cell 9: SmartConvNet Definition
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to recalibrate channel-wise features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.se(x)
        return x * scale

class BottleneckResidualBlock(nn.Module):
    """Bottleneck residual block with SE attention."""
    def __init__(self, in_channels, out_channels, downsample=False, expansion=4):
        super().__init__()
        stride = 2 if downsample else 1
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class SmartConvNet(nn.Module):
    """Enhanced CNN for seismic waveform inversion, fixing output shape to [batch_size, 1, 70, 70]."""
    def __init__(self, input_channels=5, output_size=(70, 70)):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = BottleneckResidualBlock(64, 128, downsample=True)
        self.layer2 = BottleneckResidualBlock(128, 256, downsample=True)
        self.layer3 = BottleneckResidualBlock(256, 512, downsample=True)
        self.layer4 = BottleneckResidualBlock(512, 1024, downsample=True)
        self.layer5 = BottleneckResidualBlock(1024, 1024, downsample=False)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False),  # Ensure [70, 70]
            nn.LayerNorm([70, 70])
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.decoder(x)
        return x * 1000 + 1500  # Match DumbNet scaling


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


model = SmartConvNet().to(device)


criterion = nn.L1Loss()
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)


n_epochs = 50

history = []

for epoch in range(1, n_epochs+1):
    print(f'[{epoch:02d}] Begin train')

    # Train
    model.train()
    train_losses = []
    for inputs, targets in tqdm(dltrain, desc='train', leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optim.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        optim.step()

        train_losses.append(loss.item())

    print('Train loss: {:.5f}'.format( np.mean(train_losses) ))

    # Valid
    model.eval()
    valid_losses = []
    for inputs, targets in tqdm(dlvalid, desc='valid', leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.inference_mode():
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        valid_losses.append(loss.item())
    
    print('Valid loss: {:.5f}'.format( np.mean(valid_losses)) )
    history.append({
        'train': np.mean(train_losses),
        'valid': np.mean(valid_losses)
    })

    # Plot last result
    if epoch % 4 == 0:
        y = targets[0, 0].detach().cpu()
        y_pred = outputs[0, 0].detach().cpu()
        
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
        fig.suptitle(f'Epoch {epoch} | Valid: {np.mean(valid_losses):.5f}')
        ax[0].imshow(y)
        ax[1].imshow(y_pred)
        plt.show()



pd.DataFrame(history).plot();


import csv


%%time
test_files = list(Path('/kaggle/input/waveform-inversion/test').glob('*.npy'))
len(test_files)


x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]

        return np.load(test_file), test_file.stem


ds = TestDataset(test_files)
dl = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)


# Train
model.eval()
with open('submission.csv', 'wt', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for inputs, oids_test in tqdm(dl, desc='test'):
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = model(inputs)

        y_preds = outputs[:, 0].cpu().numpy()
        
        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(
                    zip(
                        x_cols,
                        [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                    )
                )
                row['oid_ypos'] = f"{oid_test}_y_{y_pos}"
            
                writer.writerow(row)

