%%writefile config.yaml
KAGGLE_TRAIN_DIR1 : "/kaggle/input/open-wfi-1/openfwi_float16_1"
KAGGLE_TRAIN_DIR2 : "/kaggle/input/open-wfi-2/openfwi_float16_2"
KAGGLE_TEST_DIR : "/kaggle/input/open-wfi-test/test"
WORKING_DIR : "/kaggle/working"
TEST_SIZE : 0.1
BATCH_SIZE : 256
MAX_EPOCHS : 1
LEARNING_RATE : 1e-5
WEIGHT_DECAY : 1e-6
PLOT_EVERY_STEPS : 1000
READ_WEIGHTS : "/kaggle/input/classify-best0531-loss022/classify_best0531_loss022.pt"
TRAIN : "True"
TEST_WEIGHTS : "/kaggle/input/classify-best0531-loss022/classify_best0531_loss022.pt"
FACTOR : 0.8
PATIENCE : 0
ES_EPOCHS : 20
SEED : 99


import yaml

with open("config.yaml", "r") as file_obj:
    cfg = yaml.safe_load(file_obj)


import os
import numpy as np
from pathlib import Path
import datetime
import random
import time


import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# This is classification case.
LabelsMap = {
    0: "CurveFault_A",
    1: "CurveFault_B",
    2: "CurveVel_A",
    3: "CurveVel_B",
    4: "FlatFault_A",
    5: "FlatFault_B",
    6: "FlatVel_A",
    7: "FlatVel_B",
    8: "Style_A",
    9: "Style_B",
}
LabelToNum = {v: k for k, v in LabelsMap.items()}

def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

def get_train_files(data_path):
    all_inputs = [
        f for f in Path(data_path).rglob("*.npy")
        if ('seis' in f.stem) or ('data' in f.stem)
    ]

    assert all(f.exists() for f in all_inputs)

    return all_inputs

class SeismicDataset(Dataset):
    def __init__(self, inputs_files, mode, n_examples_per_file=500):
        self.inputs_files = inputs_files
        self.n_examples_per_file = n_examples_per_file
        self.mode = mode

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = os.path.basename(os.path.dirname(self.inputs_files[file_idx]))
        if y == 'data' or y == 'model':
            y = os.path.basename(os.path.dirname(os.path.dirname(self.inputs_files[file_idx])))
        y = LabelToNum[y]

        if self.mode == 'train': 
            if np.random.random() < 0.5:
                X = X[::-1, :, ::-1]
            
        try:
            return X[sample_idx].copy(), y
        finally:
            del X, y


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        test_file = self.test_files[i]
        return np.load(test_file), test_file.stem


import datetime
import random
import torch
import numpy as np

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def seed_everything(
    seed_value: int
) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d((2,2,80,80)),
            nn.Conv2d(5,5,kernel_size=(4,4),stride=(4,1),padding=(0,1)),
            nn.Conv2d(5,5,kernel_size=(4,4),stride=(4,1),padding=(0,1)),
            nn.InstanceNorm2d(5)
        )
        self.features = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),#(B,32,36,36)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),#(B,64,18,18)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),#(B,128,9,9)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*81, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


import sys
import os
import time
import gc
import numpy as np


import torch
import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from torchinfo import summary



LabelsMap = {
    0: "CurveFault_A",
    1: "CurveFault_B",
    2: "CurveVel_A",
    3: "CurveVel_B",
    4: "FlatFault_A",
    5: "FlatFault_B",
    6: "FlatVel_A",
    7: "FlatVel_B",
    8: "Style_A",
    9: "Style_B",
}
LabelToNum = {v: k for k, v in LabelsMap.items()}


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get Dataset
    all_inputs = []
    for x in [cfg["KAGGLE_TRAIN_DIR1"], cfg["KAGGLE_TRAIN_DIR2"]]:
        all_inputs_tmp = get_train_files(x)
        all_inputs.extend(all_inputs_tmp)
    print("Total number of input files:", len(all_inputs))


    train_inputs, valid_inputs = train_test_split(all_inputs, test_size=cfg["TEST_SIZE"], random_state=cfg["SEED"])
    print(f"Num of train files: {len(train_inputs)}")
    print(f"Num of valid files: {len(valid_inputs)}")

    dstrain = SeismicDataset(train_inputs, 'train')

    dltrain = DataLoader(
        dstrain,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )
    
    dsvalid = SeismicDataset(valid_inputs, 'valid')

    dlvalid = DataLoader(
        dsvalid,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )

    # Define model
    if cfg["READ_WEIGHTS"] != "None":
        print("Reading weights from:", cfg["READ_WEIGHTS"])
        model = SimpleNet()
        model.load_state_dict(torch.load(cfg["READ_WEIGHTS"], map_location='cuda', weights_only=True))
        model = model.to(device)
    else:
        model = SimpleNet().to(device)


    # Define training params
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["LEARNING_RATE"]), weight_decay=float(cfg["WEIGHT_DECAY"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=float(cfg["FACTOR"]), patience=int(cfg["PATIENCE"]))

    best_val_loss = 1000.0
    epochs_wo_improvement = 0
    t0 = time.time()

    for epoch in range(1, int(cfg["MAX_EPOCHS"])+1):
        # Train
        model.train()
        train_losses = []
        correct = 0
        for step, (inputs, targets) in enumerate(dltrain):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1).cpu().numpy()
            correct += (outputs.argmax(1) == targets).type(torch.float).sum().item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        accuracy = (correct / len(dltrain.dataset))*100
        if step % int(cfg["PLOT_EVERY_STEPS"]) == 1 or step == len(dltrain) - 1:
            trn_loss = np.mean(train_losses)
            t1 = format_time(time.time() - t0)
            lr = optimizer.param_groups[-1]['lr']
            print(
                    f"Epoch: {epoch:02d} Step {step+1}/{len(dltrain)}  Trn Loss: {trn_loss:.2f} Accuracy: {accuracy:.2f}% LR: {lr:.2e} Elapsed Time: {t1}",
                    flush=True,
                )

        
        # Valid
        model.eval()
        valid_losses = []
        correct = 0
        for inputs, targets in dlvalid:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1).cpu().numpy()
            correct += (outputs.argmax(1) == targets).type(torch.float).sum().item()
            valid_losses.append(loss.item())
        accuracy = (correct / len(dlvalid.dataset))*100


        # Gater loss on the same device
        t1 = format_time(time.time() - t0)
        trn_loss = np.mean(train_losses)
        val_loss = np.mean(valid_losses)

        free, total = torch.cuda.mem_get_info(device=0)
        mem_used = (total - free) / 1024**3

        # Log
        print(
            f"\nEpoch: {epoch:02d}  Trn Loss: {trn_loss:.2f}  Val Loss: {val_loss:.2f}  Accuracy: {accuracy:.2f}%  GPU Usage: {mem_used:.2f}GB  Elapsed Time: {t1}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_wo_improvement = 0
            torch.save(model.state_dict(), "my_best_model.pt")
            print(f"\nNew best val_loss: {val_loss:.2f}\n", flush=True)

        elif epoch == cfg["MAX_EPOCHS"]:
            torch.save(model.state_dict(), "max_epoch.pt")
        else:
            epochs_wo_improvement += 1
            print(f"\nEpochs without improvement: {epochs_wo_improvement}\n", flush=True)

        if epochs_wo_improvement == cfg["ES_EPOCHS"]:
            break

        scheduler.step(val_loss)


    # Cleanup
    del model, optimizer, scheduler
    del dltrain, dlvalid, dstrain, dsvalid
    gc.collect()
    torch.cuda.empty_cache()

    return
    
seed_everything(cfg["SEED"])
if cfg["TRAIN"] == "True":
    train(cfg)


print("==================== Output Classification Result ==================")



device = "cuda" if torch.cuda.is_available() else "cpu"

test_files = list(Path("/kaggle/input/open-wfi-test/test").glob("*.npy"))
ds = TestDataset(test_files)
dl = DataLoader(ds, batch_size=cfg["BATCH_SIZE"], num_workers=4, pin_memory=False)


print("Reading weights from:", cfg["TEST_WEIGHTS"])
model = SimpleNet()
model.load_state_dict(torch.load(cfg["TEST_WEIGHTS"], map_location='cuda', weights_only=True))
model = model.to(device)
model.eval()

result = []
for i, (inputs, oid) in enumerate(dl):
    inputs = inputs.to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)

    preds = outputs.argmax(dim=1).cpu().numpy()
    result.extend(preds)
    n = (i+1)*cfg["BATCH_SIZE"]
    if n%4096 == 0:
        print(f"processing {n} files")

filename = "test_type.txt"
with open(filename, 'w', encoding='utf-8') as f:
    for item in result:
        f.write(f"{item}\n")

print("output complete")

