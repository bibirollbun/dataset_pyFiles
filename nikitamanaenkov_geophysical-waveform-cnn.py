import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr
from scipy.ndimage import rotate


velocity = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model/model2.npy')
seismic = np.load('/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data2.npy')

print("Velocity map shape:", velocity.shape)  
print("Seismic data shape:", seismic.shape)   


def plot_shot_gather(data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    norm = np.max(np.abs(data))
    for i in range(data.shape[1]):
        trace = data[:, i] / norm  
        ax.plot(trace + i, color='black') 
    ax.set_title(title)
    ax.set_xlabel("Receiver Index (shifted)")
    ax.set_ylabel("Time")
    ax.invert_yaxis()
    plt.show()


def plot_validation_results(epoch, outputs, targets, valid_losses):
    if epoch % 4 == 0:
        y = targets[0, 0].detach().cpu()
        y_pred = outputs[0, 0].detach().cpu()

        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
        fig.suptitle(f'Epoch {epoch} | Valid: {np.mean(valid_losses):.5f}')
        ax[0].imshow(y, cmap='viridis')
        ax[1].imshow(y_pred, cmap='viridis')
        plt.show()  


def plot_seismic_waveform(seismic, sample, source_idx, receiver_idx):
    waveform = seismic[sample, source_idx, :, receiver_idx]  

    plt.figure(figsize=(12, 4))
    plt.plot(waveform, lw=0.8)
    plt.title(f"Seismic Waveform - Sample {sample}, Source {source_idx}, Receiver {receiver_idx}")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_velocity_map(velocity, sample):
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(velocity[sample, 0], cmap='jet', origin='upper')
    plt.colorbar(img, ax=ax, label="Velocity (km/s)")
    ax.set_title(f"Velocity Map - Sample {sample}")
    ax.set_xlabel("Horizontal Position (x)")
    ax.set_ylabel("Depth (z)")
    plt.show()


def plot_seismic_data(seis, sample_id):
    plt.figure(figsize=(10, 6))
    plt.title(f"Seismic Data - Batch 0, Source 0")
    plt.imshow(seis[0, 0], aspect='auto', cmap='seismic')
    plt.colorbar(label="Amplitude")
    plt.xlabel("Receivers")
    plt.ylabel("Timesteps")
    plt.show()


sample = 11
plot_shot_gather(seismic[sample, 0], f"Shot Gather - Sample {sample}, Source 0")
plot_seismic_waveform(seismic, sample, 0, 35)
plot_velocity_map(velocity, sample)
plot_seismic_data(seismic, sample)  



def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

all_inputs = [
    f
    for f in Path('/kaggle/input/waveform-inversion/train_samples').rglob('*.npy')
    if ('seis' in f.stem) or ('data' in f.stem)
]

all_outputs = inputs_files_to_output_files(all_inputs)
assert all(f.exists() for f in all_outputs)

train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)]  
valid_inputs = [f for f in all_inputs if f not in train_inputs]
train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)


class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500, augmentation_prob=0.3, crop_size=None, rotate_prob=0.5, flip_prob=0.5, normalize=True):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file
        self.augmentation_prob = augmentation_prob
        self.crop_size = crop_size
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.normalize = normalize  

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        X_sample = X[sample_idx].copy()

        if np.random.rand() < self.augmentation_prob:
            noise = np.random.normal(0, 0.01, X_sample.shape)
            X_sample += noise
        
        if self.crop_size is not None:
            X_sample = self.random_crop(X_sample, self.crop_size)

        if np.random.rand() < self.rotate_prob:
            X_sample = self.random_rotate(X_sample)

        if np.random.rand() < self.flip_prob:
            X_sample = self.random_flip(X_sample)

        if self.normalize:
            X_sample = (X_sample - np.mean(X_sample)) / np.std(X_sample)  

        return X_sample.copy(), y[sample_idx].copy()

    def random_crop(self, X_sample, crop_size):
        h, w = X_sample.shape
        crop_h, crop_w = crop_size
        top = np.random.randint(0, h - crop_h)
        left = np.random.randint(0, w - crop_w)
        cropped = X_sample[top:top+crop_h, left:left+crop_w]
        return cropped

    def random_rotate(self, X_sample):
        angle = np.random.uniform(-45, 45)
        rotated = rotate(X_sample, angle, mode='nearest', reshape=False)  
        return rotated

    def random_flip(self, X_sample):
        flip_choice = np.random.choice([0, 1])  
        if flip_choice == 0:
            X_sample = np.fliplr(X_sample)  
        else:
            X_sample = np.flipud(X_sample)  
        return X_sample


dstrain = SeismicDataset(train_inputs, train_outputs)
dltrain = DataLoader(dstrain, batch_size=64, shuffle=True, pin_memory=True, drop_last=True, num_workers=4, persistent_workers=True)

dsvalid = SeismicDataset(valid_inputs, valid_outputs)
dlvalid = DataLoader(dsvalid, batch_size=64, shuffle=False, pin_memory=True, drop_last=False, num_workers=4, persistent_workers=True)



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SeismicModel(nn.Module):
    def __init__(self):
        super(SeismicModel, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 70 * 70)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.avgpool(x)  
        x = x.view(x.size(0), -1)  

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(x.size(0), 1, 70, 70)
        return x


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  
        self.delta = delta        
        self.counter = 0          
        self.best_loss = None     
        self.early_stop = False   

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train(model, train_dataloader, valid_dataloader, epochs=10, patience=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    valid_losses = []
    ssim_scores = []
    corr_scores = []
    learning_rates = [] 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(valid_dataloader, desc=f'Validating Epoch {epoch+1}'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_losses.append(avg_valid_loss)

        target_np = targets[0, 0].detach().cpu().numpy()
        output_np = outputs[0, 0].detach().cpu().numpy()
        ssim_score = ssim(target_np, output_np, data_range=output_np.max() - output_np.min())
        corr_score, _ = pearsonr(target_np.flatten(), output_np.flatten())

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        ssim_scores.append(ssim_score)
        corr_scores.append(corr_score)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.5f}, Valid Loss: {avg_valid_loss:.5f}")
        print(f"          SSIM: {ssim_score:.4f}, Pearson Corr: {corr_score:.4f}")

        scheduler.step(avg_valid_loss)
        
        if early_stopping(avg_valid_loss):
            print("Early stopping triggered.")
            break
            
        plot_validation_results(epoch, outputs, targets, valid_losses)

    plt.figure(figsize=(16,5))

    plt.subplot(1,4,1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Validation Loss')
    plt.legend()

    plt.subplot(1,4,2)
    plt.plot(range(1, epochs+1), ssim_scores, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')

    plt.subplot(1,4,3)
    plt.plot(range(1, epochs+1), corr_scores, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Corr over Epochs')

    plt.subplot(1,4,4)
    plt.plot(range(1, epochs+1), learning_rates, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')

    plt.tight_layout()
    plt.show()




class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        test_file = self.test_files[i]
        return np.load(test_file), test_file.stem

x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols

test_files = [f for f in Path('/kaggle/input/waveform-inversion/test').rglob('*.npy')]

ds_test = TestDataset(test_files)
dl_test = DataLoader(ds_test, batch_size=8, num_workers=4, pin_memory=True)

def test_and_save_results(model, test_dataloader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('submission.csv', 'wt', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for inputs, oids_test in tqdm(test_dataloader, desc='Testing'):
            inputs = inputs.to(device)
            with torch.no_grad():
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SeismicModel()
train(model, dltrain, dlvalid, epochs=50)  
test_and_save_results(model, dl_test)

