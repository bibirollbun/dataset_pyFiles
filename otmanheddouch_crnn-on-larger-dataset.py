import numpy as np
import pandas as pd 
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

import torch 
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import train_test_split
from pathlib import Path
import csv

torch.cuda.empty_cache()

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = "models/"
os.makedirs(OUTPUT_DIR, exist_ok =True)

MODEL_PATH = "models/model.pth"
TRAIN_DIRS = ["/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72"]
BATCH_SIZE = 32 # TODO CHANGE LATER 
SEED = 42
N_EXAMPLES_PER_FILE = 500

INPUT_SHAPE = (1, 5, 72, 72)
OUTPUT_SHAPE = (1, 70, 70)
TEST_PATH ="/kaggle/input/waveform-inversion/test"
TRAIN_RATIO = 0.8

lr = 3e-4 
weight_decay = 0.00001
EPOCHS = 3
EARLY_STOPPING_EPOCH = 3 # the number of epochs to wait for improvement in the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Auto-detect GPU
print("Device:", device)
cuda_count = torch. cuda. device_count()
print(f"We have {cuda_count} cuda devices")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



random_style = "FlatFault_A"
random_style



loaded_example_input = np.load(os.path.join(os.path.join(TRAIN_DIRS[0], random_style), "seis4_1_0.npy"))
loaded_example_output = np.load(os.path.join(os.path.join(TRAIN_DIRS[0], random_style), "vel4_1_0.npy"))


loaded_example_input.shape, loaded_example_output.shape


sample_number = 14
sample_y = loaded_example_output[sample_number]
print("Velocity shape:", sample_y.shape)

fig, ax = plt.subplots(1, figsize=(9, 6))
im = ax.imshow(sample_y[0], cmap="jet")
ax.set_xticks(range(0, 70, 10))
ax.set_xticklabels(range(0, 700, 100))
ax.set_yticks(range(0, 70, 10))
ax.set_yticklabels(range(0, 700, 100))

ax.set_xlabel("Offset")
ax.set_ylabel("Depth")

plt.colorbar(im).ax.set_title("km/s")



sample_number = 14

sample_x = loaded_example_input[sample_number]
print("Data shape:", sample_x.shape)
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, ax in enumerate(axes):
   ax.imshow(sample_x[i], cmap="seismic", aspect="auto", vmin=-0.5,vmax=0.5)

   ax.set_ylabel('Time (ms)', fontsize=16)
   ax.set_xlabel('Signals', fontsize=16)
plt.show()

sample_y = loaded_example_output[sample_number]
print("Velocity shape:", sample_y.shape)

fig, ax = plt.subplots(1, figsize=(9, 6))
im = ax.imshow(sample_y[0], cmap="jet")
ax.set_xticks(range(0, 70, 10))
ax.set_xticklabels(range(0, 700, 100))
ax.set_yticks(range(0, 70, 10))
ax.set_yticklabels(range(0, 700, 100))

ax.set_xlabel("Offset")
ax.set_ylabel("Depth")

plt.colorbar(im).ax.set_title("km/s")

plt.show()


class FWIDataset(Dataset):
    def __init__(self, input_files, output_files , n_examples_per_files, data_transform=None, label_transform=None):
        self.input_files = input_files
        self.output_files = output_files
        self.n_examples_per_files = n_examples_per_files
        self.data_transform = data_transform
        self.label_transform = label_transform
        
    def __getitem__(self, idx):
        
        # get the file index and the sample index
        batch_idx, sample_idx = idx // self.n_examples_per_files, idx % self.n_examples_per_files
        # check if it is exists
        if batch_idx >= len(self.input_files):
            raise IndexError("File doasn't exists")
        # load the files
        data = np.load(self.input_files[batch_idx], mmap_mode='r')
        label = np.load(self.output_files[batch_idx], mmap_mode='r')
        # load the exact sample
        seism_data = data[sample_idx].copy().astype('float32')
        velocity_data = label[sample_idx].copy().astype('float32')

        # remove the variables from the memory
        del data, label
        
        if self.data_transform:
           seism_data = self.data_transform(seism_data)
        if self.label_transform:
            velocity_data = self.label_transform(velocity_data)
        return torch.from_numpy(seism_data) ,torch.from_numpy(velocity_data)

    def __len__(self):
        return self.n_examples_per_files * len(self.input_files)


def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)

class LogTransform(object):
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)
 

data_transform = Compose([
    LogTransform(),
])

label_transform = Compose([
    LogTransform(),
])


output_files = []
input_files = []
for train_dir in TRAIN_DIRS:
    for dirname, _, filenames in os.walk(train_dir):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if "model" in filename or "vel" in filename:
                output_files.append(path)
            elif "csv" not in filename:
                input_files.append(path)

print("Length of input files: ", len(input_files), " Length of output files: ", len(output_files))



train_input_files, val_input_files, train_output_files, val_output_files = train_test_split(input_files,  output_files, test_size=0.2, shuffle=True, random_state=42)
print(f"Length of training: {len(train_input_files)}, Length of validation : {len(val_input_files)}")


train_fwi_dataset = FWIDataset(train_input_files, train_output_files, N_EXAMPLES_PER_FILE, 
                         # data_transform=data_transform, 
                               # label_transform=label_transform
                              )

val_fwi_dataset = FWIDataset(val_input_files, val_output_files, N_EXAMPLES_PER_FILE, 
                         # data_transform=data_transform, 
                             #label_transform=label_transform
                            )
print(f"Length of training dataset : {len(train_fwi_dataset)}, Length of validation dataset : {len(val_fwi_dataset)}")


train_random_sampler = RandomSampler(train_fwi_dataset)
val_random_sampler = RandomSampler(val_fwi_dataset)

train_fwi_dataloader = DataLoader(train_fwi_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4, sampler=train_random_sampler)
val_fwi_dataloader = DataLoader(val_fwi_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4, sampler=val_random_sampler)


class _BidirectionalLSTM(nn.Module):
  def __init__(self, input_size: int, hidden_size: int, output_size: int):
    super(_BidirectionalLSTM, self).__init__()
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first =True)
    self.linear = nn.Linear(in_features = hidden_size* 2, out_features=output_size)
  def forward(self, x: torch.Tensor)-> torch.Tensor:

    recurrent, _ = self.lstm(x)
    seq_lenght, batch_size, inputs_size = recurrent.size()
    seq_lenght2 = recurrent.reshape(seq_lenght * batch_size, inputs_size)

    out = self.linear(seq_lenght2)
    out = out.reshape(seq_lenght, batch_size, -1)
    return out

# (batch_size, 5, 72, 72) => (batch_size, 70, 70)
class CRNN(nn.Module):
  def __init__(self, in_channels: int, output_size:int):
    super(CRNN, self).__init__()
    self.conv_layer = nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(0.3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(0.3),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(512),
        # nn.ReLU(0.2),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
        # nn.ReLU(0.2),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(512),
        # nn.ReLU(0.2),

    )
    self.output_size = output_size
    self.recurrent_layer = nn.Sequential(
        _BidirectionalLSTM(512, 256, 256),
        _BidirectionalLSTM(256, 256,  256),
    )
    self.fcl =  nn.Sequential(
        nn.Linear(in_features = 14592, out_features=516) ,
        nn.Linear(in_features = 516, out_features=self.output_size*self.output_size) ,
        
    )   
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      batch_size = x.shape[0]
      x = x.float()
      mean = torch.mean(x, dim=(2, 3), keepdim=True)
      std = torch.std(x, dim=(2, 3), keepdim=True)
      x_norm = (x - mean) / (std + 1e-8) # Epsilon for numerical stability
      
      features = self.conv_layer(x_norm) # squeze the first [32, 512, 3, 19]

      features = features.reshape(batch_size, 512, 57) # torch.Size([batch_size, 512, 57])
      features = features.permute(0, 2, 1) # chagnge the dimension
      recurrent = self.recurrent_layer(features) 
      flattened = recurrent.reshape(batch_size, -1) # flatten
      linear_features = self.fcl(flattened)
      output = linear_features.reshape(batch_size, 1, self.output_size, self.output_size) # reshape to (batch_size, 1, output_size, output_size)
      output = output * 1000.0 + 1500.0
      return output 


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.to(device)
    model.train()
    total_loss = 0
    progress = tqdm(dataloader, desc="Training Epoch")
    for data, labels in progress:
        optimizer.zero_grad()
        data = data.to(device).float()
        labels = labels.to(device).float()

        with torch.autocast(device_type=str(device)):
            outputs = model(data)
            loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():3f}')

    return total_loss / len(dataloader)


def val(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    progress = tqdm(dataloader, desc="Validation")
    i = 0 # index to tracking storing the validation output of the first output
    first_output = None
    first_label = None
    with torch.no_grad():
        for data, labels in progress:
            data = data.to(device).float()
            labels = labels.to(device).float()

            outputs = model(data)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            progress.set_postfix(loss=f'{loss.item():3f}')
            if i== 0:
                first_output = outputs[0].detach().cpu()
                first_label = labels[0].detach().cpu()
            
    return total_loss / len(dataloader), first_output, first_label

    


 torch.cuda.empty_cache()
gpu_devices = ','.join([str(id) for id in range(0, cuda_count)])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
# define the model
network = CRNN(5, 70)
# load the model into all cuda available (in our case 2)
netowrk = nn.DataParallel(network)
network.to(device)

# Apply initialization
# define the Adam optmizer 
optimizer = torch.optim.AdamW( network.parameters(), lr=lr, weight_decay=weight_decay)
# scheduler 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
# define the loss
loss_fn = nn.L1Loss()


def plot_output(ground_truth, ground_output):
    values = [ground_truth, ground_output]
    titles = ["Prediction", "Ground Output"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    for i in range(0, 2):
        im = axes[i].imshow(values[i], cmap="jet")
        axes[i].set_xticks(range(0, 70, 10))
        axes[i].set_xticklabels(range(0, 700, 100))
        axes[i].set_yticks(range(0, 70, 10))
        axes[i].set_yticklabels(range(0, 700, 100))
        
        axes[i].set_xlabel("Offset")
        axes[i].set_ylabel("Depth")
        axes[i].set_title(titles[i])
    plt.show()


 torch.cuda.empty_cache()
losses = {
    "val_loss":[],
    "train_loss":[]
}
best_loss = 10000
epoch_waited = 0
for epoch in range(0, 10):
    torch.cuda.empty_cache()
    print(f"Training for {epoch} epoch")
    avg_loss = train_one_epoch(network, train_fwi_dataloader, loss_fn, optimizer, device)
    avg_loss_val, val_output, val_label = val(network,loss_fn, val_fwi_dataloader , device)
    scheduler.step(avg_loss_val)
    # Printing the training and validation loss
    print(f"==Training loss:{avg_loss} Validation loss:{avg_loss_val}===")
    # Ploting every 5 epoch the results
    if epoch % 5 == 0:
        plot_output(val_output[0],val_label[0])
    # Append the loss for later ploting
    losses["train_loss"].append([avg_loss, i])
    losses["val_loss"].append([avg_loss_val, i])
    if best_loss > avg_loss_val:
        best_loss = avg_loss_val
        print(f"Saving the best model to {MODEL_PATH}")
        torch.save(network.state_dict().copy(), MODEL_PATH)
    else:
        epoch_waited += 1
    # break if no improvement happen
    if epoch_waited >= EARLY_STOPPING_EPOCH:
        print("Breaking... No improvement")
        break


network.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
network.eval()


avg_loss_val, val_output, val_label = val(network,loss_fn, val_fwi_dataloader , device)
print("Loss:", avg_loss_val)


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def _preprocess(x):
    x = F.interpolate(x, size=(70, 70), mode='area')
    x = F.pad(x, (1,1,1,1), mode='replicate')
    return x

def _helper(x, ):
    before_shape = x.shape
    before_mem = x.nbytes / 1e6
    x = torch.from_numpy(x).float()

    # Interpolate and pad
    x = _preprocess(x)
    x = x.reshape(5, 72, 72)
    return x
    
class TestDataset(Dataset):
    def __init__(self, files):
        self.files = files


    def __len__(self):
        return len(self.files)


    def __getitem__(self, i):
        test_file = self.files[i]
        x = _helper(np.load(test_file).reshape(-1, 5, 1000, 70))

        return x, test_file.stem


test_files = list(Path(TEST_PATH).glob("*.npy"))
test_dataset = TestDataset(test_files)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=False, num_workers=4)



x_cols = [f"x_{i}" for i in range(1, 70, 2)]
fieldnames = ["oid_ypos"] + x_cols

with open("submission.csv", "wt", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for inputs, oids_test in tqdm(test_dataloader, desc="Testing data"):
        inputs = inputs.to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                outputs = network(inputs)

        y_preds = outputs[:, 0].cpu().numpy()

        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(zip(x_cols, [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]))
                row["oid_ypos"] = f"{oid_test}_y_{y_pos}"

                writer.writerow(row)





