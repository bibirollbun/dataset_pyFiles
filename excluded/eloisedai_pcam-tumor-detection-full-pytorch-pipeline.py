import typing as tp
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import ToTensor 
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter


from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device", device)


!pip install kaggle
creds = '{"username":"xxxxx","key":"xxxxx"}'
from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)


import os
import zipfile

root = "/kaggle/input/"
dataset_dir = "/kaggle/input/histopathologic-cancer-detection"
zip_file = "histopathologic-cancer-detection.zip"
train_path = os.path.join(dataset_dir, "train")

if not os.path.exists(root):
  os.mkdir(root)

if not os.path.exists('results'):
  os.mkdir('results')

if not os.path.exists(train_path):
    print("Downloading Histopathologic Cancer Detection dataset...")
    !kaggle competitions download -c histopathologic-cancer-detection -p {root} --force
else:
    print("Dataset zip already downloaded.")

if not os.path.exists(train_path):
    print("Unzipping dataset...")
    with zipfile.ZipFile(os.path.join(root, zip_file), 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
else:
    print("Dataset already unzipped.")


from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

class PcamDatasetKaggle(torchvision.datasets.VisionDataset):
    def __init__(self, root, split, transform, target_transform = None):
         super().__init__(root, transform=transform, target_transform=target_transform)
         self.root = root
         self.split = split
         self.transform = transform
         self.img_path = os.path.join(self.root, "train")

         self.full_labels = pd.read_csv(self.root+'/train_labels.csv')
         X_train, X_test, y_train, y_test = train_test_split(self.full_labels['id'],
                                                             self.full_labels['label'],
                                                             test_size = 0.2, 
                                                             train_size = 0.8,
                                                             random_state=42,
                                                             shuffle=True,
                                                             stratify=self.full_labels['label'])
        
         if (split == "train"):
             self.imgs = X_train + ".tif"
             self.labels = y_train
         elif (split == "val"):
             self.imgs = X_test + ".tif"
             self.labels = y_test
         else:
             self.img_path = os.path.join(self.root, self.split)
             self.imgs = pd.Series(list(sorted(os.listdir(self.img_path))))
             self.labels = pd.Series(torch.full((len(self.imgs),), -10))      
         assert len(self.labels) == len(self.imgs)
         print("Split", split, "Negative/Positive samples % " , 100.0*(self.labels.value_counts() / self.labels.shape[0]))

    def __getitem__(self, idx):
        assert idx < len(self.imgs)
        img = Image.open(os.path.join(self.img_path, self.imgs.iloc[idx]))
        if self.transform:
            img = self.transform(image = np.array(img))
        label = self.labels.iloc[idx]
        return img['image'].to(torch.float32), label
    def __len__(self) :
        return len(self.imgs)

def check_dataset_leakage(dataset1, dataset2):
    duplicates = set(dataset1.imgs) & set(dataset2.imgs)
    assert len(duplicates) == 0
    
def check_same_imgs(dataset1, dataset2):
    duplicates = set(dataset1.imgs) & set(dataset2.imgs)
    assert len(duplicates) == len(dataset1.imgs)
    assert len(duplicates) == len(dataset2.imgs)


import torchvision.transforms as transforms

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Preprocess images with transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)), #Match resnet original input size            
    transforms.ToTensor()
])

transform_data_augment = transforms.Compose([ 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size = (5,5),sigma=(0.2, 0.7)),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(
        brightness=0.4, 
        contrast=0.4, 
        saturation=0.1, 
        hue=0.03
    ),
    transforms.RandomResizedCrop(size = (224, 224), scale = (0.7, 1.0)),
    transforms.ToTensor()
])




!pip install albumentations


import albumentations as A
transform_data_augment = A.Compose([A.Resize(224, 224), 
                    A.HorizontalFlip(), 
                    A.VerticalFlip(), 
                    A.RandomRotate90(), 
                    A.Transpose(), 
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=60, p=.75),
                    A.OpticalDistortion(),
                    A.GridDistortion(), 
                    A.RandomBrightnessContrast(p=0.3), 
                    A.RandomGamma(p=0.3), 
                    A.OneOf([A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), 
                            A.ChannelShuffle(p=0.3), A.CLAHE(p=0.3)]),
                    A.Normalize(normalization="image_per_channel", p=1.0),
                              A.ToTensorV2()])

transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(normalization="image_per_channel", p=1.0),
                A.ToTensorV2()
            ])


from copy import deepcopy

""" PCAM pytorch version but the dataset is not clean 
training_set_original = datasets.PCAM(root="/kaggle/input", split="train",download = True, transform = transform) 
training_set_augment = datasets.PCAM(root="/kaggle/input", split="train",download = True, transform = transform_data_augment)
val_set = datasets.PCAM(root="/kaggle/input", split="val", download=True, transform = transform)
test_set = datasets.PCAM(root="/kaggle/input", split="test", download=True, transform = transform)
"""

training_set_original = PcamDatasetKaggle(root=dataset_dir, split="train", transform = deepcopy(transform)) 
training_set_augment = PcamDatasetKaggle(root=dataset_dir, split="train", transform = deepcopy(transform_data_augment)) 

val_set = PcamDatasetKaggle(root=dataset_dir, split="val", transform = deepcopy(transform)) 
val_set_augment = PcamDatasetKaggle(root=dataset_dir, split="val", transform = deepcopy(transform_data_augment)) 

test_set = PcamDatasetKaggle(root=dataset_dir, split="test", transform = deepcopy(transform))
test_set_augment = PcamDatasetKaggle(root=dataset_dir, split="test", transform = deepcopy(transform_data_augment)) #For TTA

check_dataset_leakage(training_set_original, val_set)
check_dataset_leakage(training_set_original, test_set)
check_dataset_leakage(val_set, test_set)
check_same_imgs(training_set_original, training_set_augment)
check_same_imgs(val_set, val_set_augment)
check_same_imgs(test_set, test_set_augment)


import matplotlib.pyplot as plt

def plot_training_set_sample(training_set, 
                             file_name = "results/pcam/data.png", 
                             rows = 5, 
                             cols = 5, 
                             mean_stdev = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])):
    mean = mean_stdev[0].numpy()
    std  = mean_stdev[1].numpy()
    fig = plt.figure(figsize=(2*cols, 2*rows))
    for i in range(1, rows*cols + 1):
        random_idx = torch.randint(len(training_set), (1,)).item()
        fig.add_subplot(rows, cols, i)
        img = training_set[random_idx][0].permute(1,2,0).numpy() 
        img_unnormalized = img*std + mean
        img_unnormalized = np.clip(img_unnormalized, 0, 1)
        plt.imshow(img_unnormalized)
        plt.axis("off")
        plt.title(training_set[random_idx][1])
    plt.savefig(file_name)
    plt.show()
    


import os
from datetime import datetime
exp_dir = "results/pcam/"+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
os.makedirs(exp_dir)


print("Original Training Set")
plot_training_set_sample(training_set_original, exp_dir + "/training_set_original.png",rows=2, cols=5)


print("Augmented Training Set")
plot_training_set_sample(training_set_augment, exp_dir + "/training_set_augment.png",rows=2, cols=5)


# Create DataLoader
batch_size = 128
training_set_original = PcamDatasetKaggle(root=dataset_dir, split="train", transform = deepcopy(transform)) 
training_set_augment = PcamDatasetKaggle(root=dataset_dir, split="train", transform = deepcopy(transform_data_augment)) 
val_set = PcamDatasetKaggle(root=dataset_dir, split="val", transform = deepcopy(transform)) 
val_set_augment = PcamDatasetKaggle(root=dataset_dir, split="val", transform = deepcopy(transform_data_augment)) 
test_set = PcamDatasetKaggle(root=dataset_dir, split="test", transform = deepcopy(transform))


# Create Augmented Training Dataset
training_set = ConcatDataset([training_set_original, training_set_augment])

# Create Final DataLoaders
training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6, persistent_workers = True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6, persistent_workers = True)
val_dataloader_augment = DataLoader(val_set_augment, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6, persistent_workers = True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6, persistent_workers = True)



print("Full Training Set Normalized")
plot_training_set_sample(training_set, exp_dir + "/training_set_final.png", rows = 2, cols = 5)


def compute_metrics(full_y: torch.Tensor, 
                    full_logits: torch.Tensor,  
                    full_pred: torch.Tensor,  
                    sk_learn_metrics_logits: tp.List[tp.Callable],
                    sk_learn_metrics_pred: tp.List[tp.Callable]) -> tp.Dict:
    full_y = full_y.detach().cpu().numpy()
    full_logits = torch.sigmoid(full_logits).detach().cpu().numpy()
    full_pred = full_pred.detach().cpu().numpy()
    
    results = {}
    for metric in sk_learn_metrics_logits:
        results[metric.__name__] = metric(full_y, full_logits)
    for metric in sk_learn_metrics_pred:
        results[metric.__name__] = metric(full_y, full_pred)
    return results


def run_one_epoch(model : nn.Module, 
                   training_dataloader: DataLoader,
                   optimizer: Optimizer,
                   loss_function: nn.Module,
                   scheduler : LRScheduler,
                   device: torch.cuda.device,
                   writer: SummaryWriter,
                   epoch: int,
                   sk_learn_metrics_logits: tp.List[tp.Callable],
                   sk_learn_metrics_pred: tp.List[tp.Callable],
                   threshold: float = 0.5):
    running_loss = 0.0
    num_batch = len(training_dataloader)
    full_y = torch.Tensor([]).to(device)
    full_logits = torch.Tensor([]).to(device)
    full_pred = torch.Tensor([]).to(device)
    
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    for batch, (X, y) in enumerate(training_dataloader):
        optimizer.zero_grad()
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(X).squeeze()
            loss = loss_function(logits, y.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > threshold).float()
            full_y = torch.cat([full_y, y])
            full_logits = torch.cat([full_logits, logits])
            full_pred = torch.cat([full_pred, preds])
         
        running_loss += loss.item()
        avg_loss = running_loss / (batch + 1.)
        if batch % 250 == 0:
            writer.add_scalar('Training Loss(avg)', avg_loss, batch + epoch*num_batch)
            writer.add_scalar('Training Loss (raw)', loss.item(), batch + epoch*num_batch)
    scheduler.step()
    writer.flush()
    return compute_metrics(full_y, full_logits, full_pred, sk_learn_metrics_logits, sk_learn_metrics_pred)


def eval_model(model: nn.Module,
               dataloader: DataLoader, 
               sk_learn_metrics_logits: tp.List[tp.Callable],
               sk_learn_metrics_pred: tp.List[tp.Callable],
               device: torch.cuda.device,
               threshold: float = 0.5) -> tp.Dict:
    
    model.eval()
    full_y = torch.Tensor([]).to(device)
    full_logits = torch.Tensor([]).to(device)
    full_pred = torch.Tensor([]).to(device)
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X).squeeze()
            preds = (torch.sigmoid(logits) > threshold).float()

            full_y = torch.cat([full_y, y])
            full_logits = torch.cat([full_logits, logits])
            full_pred = torch.cat([full_pred, preds])
    return compute_metrics(full_y, full_logits, full_pred, sk_learn_metrics_logits, sk_learn_metrics_pred)


import threading 
import tensorboard
from tensorboard import program

def start_tensorboard(logdir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")

# Replace 'logs' with your actual log directory
logdir = exp_dir
tb_thread = threading.Thread(target=start_tensorboard, args=(logdir,), daemon=True)
#tb_thread.start() #if you are on a local run uncomment to start tensorboard


from PIL import Image

def load_image(path):
    img = Image.open(path)
    # Convert to numpy array and add batch dimension (C, H, W)
    img_array = np.array(img)
    if len(img_array.shape) == 2:  # Grayscale image
        img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)
    else:  # Color image
        img_array = img_array.transpose(2, 0, 1)  # (C, H, W)
    return img_array
    
writer = SummaryWriter(exp_dir + '/tensorboard')
writer.add_image('training_set_original', load_image(exp_dir + "/training_set_original.png"), 0)
writer.flush()
writer.add_image('training_set_augment',  load_image(exp_dir + "/training_set_augment.png"), 0)
writer.flush()
writer.add_image('training_set_final',  load_image(exp_dir + "/training_set_final.png"), 0)
writer.flush()


from torchvision.models import densenet201, DenseNet201_Weights
model = densenet201(weights=DenseNet201_Weights.DEFAULT)

for params in model.parameters():
    params.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.7),
    nn.Linear(model.classifier.in_features, 512, bias= True),
    nn.Dropout(0.5),
    nn.Linear(512, 1, bias= True))

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

def custom_lr_find(model : nn.Module, 
                   dataloader: DataLoader,
                   loss_function: nn.Module,
                   device: str,
                   start_lr = 1e-7,
                   end_lr = 1.0,
                   num_iteration = 200):
    rates = []
    lossses = []
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=start_lr)

    
    def lr_lambda(iteration):
        return (end_lr / start_lr) ** (iteration / num_iteration)
        
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    initial_weights = model.state_dict()
    model.train()
    
    X_full = torch.Tensor([]).to(device)
    y_full = torch.Tensor([]).to(device)
    
    for h in range (0, 5):
        X, y = next(iter(dataloader))
        X = X.to(device)
        y = y.to(device)
        X_full = torch.cat([X_full, X])
        y_full = torch.cat([y_full, y])
    
    for i in range(0, num_iteration):
        optimizer.zero_grad()

        pred = model(X_full).squeeze()
        loss = loss_function(pred, y_full.float())
        lossses.append(loss.item())
        rates.append(scheduler.get_last_lr()[0])
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.load_state_dict(initial_weights)
        if(scheduler.get_last_lr()[0] > end_lr):
            break
    return rates, lossses
        
def plot_lr_find(rates, losses, file_name):
    fig = plt.Figure()
    plt.plot(rates, losses)
    plt.xscale('log')
    plt.xlabel('learning_rate')
    plt.ylabel('loss')
    plt.ylim(0.0, 1.0)
    plt.title('lr_find_results')
    plt.legend()
    plt.savefig(file_name)
    plt.figure()
    
rates, losses = custom_lr_find(model, training_dataloader, torch.nn.BCEWithLogitsLoss(), device)
plot_lr_find(rates, losses, exp_dir + '/lr_find.jpg')
writer.add_image('lr_find', load_image(exp_dir + "/lr_find.jpg"), 0)
writer.flush()


from torchvision.models import densenet201, DenseNet201_Weights, densenet121, DenseNet121_Weights
model = densenet201(weights=DenseNet201_Weights.DEFAULT)

for params in model.parameters():
    params.requires_grad = False

#Replace the last layer (to output a 1d prediction)
model.classifier = nn.Sequential(
    nn.Dropout(0.7),
    nn.Linear(model.classifier.in_features, 512, bias= True),
    nn.Dropout(0.5),
    nn.Linear(512, 1, bias= True))

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)


#optionnaly load from checkpoint
'''
model = torch.load('results/pcam/19_06_2025_11_08_15/model_'+str(5)+'.pt', weights_only = False)
for params in model.parameters():
    params.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
model = model.to(device)
'''


lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
loss_func = torch.nn.BCEWithLogitsLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)


from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report
import time
epoch_num = 2
sk_learn_metrics_logits = [roc_auc_score]
sk_learn_metrics_pred = [f1_score, accuracy_score]
for i in range(0, epoch_num):
    start_time = time.time()
    train_res = run_one_epoch(model,
                  training_dataloader,
                  optimizer,
                  loss_func,
                  scheduler,
                  device,
                  writer,
                  i,
                  sk_learn_metrics_logits,
                  sk_learn_metrics_pred)
    end_time = time.time()
    print("epoch nÂ°: ", i, " training time : ", end_time-start_time, " sec")
    start_time = time.time()
    val_res = eval_model(model, val_dataloader, sk_learn_metrics_logits, sk_learn_metrics_pred, device)
    for key in train_res.keys():
        writer.add_scalars(key, {"Train " + key: train_res[key], "Val "+ key : val_res[key]}, i*len(training_dataloader))
    end_time = time.time()
    print("epoch nÂ°: ", i, " evaluation time : ", end_time-start_time, " sec")
    torch.save(model, exp_dir+"/model_" + str(i) + ".pt")



'''
for name, param in model.features.denseblock4.denselayer32.conv1.named_parameters():
    param.requires_grad = True
    
for name, param in model.features.denseblock4.denselayer32.conv2.named_parameters():
    param.requires_grad = True
'''


# Unfreeze last two blocks (features.6 and features.7)
'''
lr = 1e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_func = torch.nn.BCEWithLogitsLoss()
# Use lower LR for fine-tuning
optimizer = torch.optim.Adam([
    {"params": model.classifier.parameters(), "lr": 1e-4},
     {"params": model.features.denseblock4.denselayer32.conv1.parameters(), "lr": 1e-5},
     {"params": model.features.denseblock4.denselayer32.conv2.parameters(), "lr": 1e-5},
 ])
'''


'''
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report
import time
sk_learn_metrics_logits = [roc_auc_score]
sk_learn_metrics_pred = [f1_score, accuracy_score]
epoch_num = 2
finetune_epoch_num = 6
for i in range(epoch_num, epoch_num + finetune_epoch_num):
    start_time = time.time()
    train_res = run_one_epoch(model,
                  training_dataloader,
                  optimizer,
                  loss_func,
                  scheduler,
                  device,
                  writer,
                  i,
                  sk_learn_metrics_logits,
                  sk_learn_metrics_pred)
    end_time = time.time()
    print("epoch nÂ°: ", i, " training time : ", end_time-start_time, " sec")
    start_time = time.time()
    val_res = eval_model(model, val_dataloader, sk_learn_metrics_logits, sk_learn_metrics_pred, device)
    for key in train_res.keys():
        writer.add_scalars(key, {"Train " + key: train_res[key], "Val "+ key : val_res[key]}, i*len(training_dataloader))
    end_time = time.time()
    print("epoch nÂ°: ", i, " evaluation time : ", end_time-start_time, " sec")
    torch.save(model, exp_dir+"/model_" + str(i) + ".pt")

'''


for params in model.parameters():
    params.requires_grad = True

lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
loss_func = torch.nn.BCEWithLogitsLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)


from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report
import time
sk_learn_metrics_logits = [roc_auc_score]
sk_learn_metrics_pred = [f1_score, accuracy_score]
epoch_num = 2
finetune_epoch_num = 5
    
for i in range(epoch_num, epoch_num + finetune_epoch_num):
    start_time = time.time()
    train_res = run_one_epoch(model,
                  training_dataloader,
                  optimizer,
                  loss_func,
                  scheduler,
                  device,
                  writer,
                  i,
                  sk_learn_metrics_logits,
                  sk_learn_metrics_pred)
    end_time = time.time()
    print("epoch nÂ°: ", i, " training time : ", end_time-start_time, " sec")
    start_time = time.time()
    val_res = eval_model(model, val_dataloader, sk_learn_metrics_logits, sk_learn_metrics_pred, device)
    for key in train_res.keys():
        writer.add_scalars(key, {"Train " + key: train_res[key], "Val "+ key : val_res[key]}, i*len(training_dataloader))
    end_time = time.time()
    print("epoch nÂ°: ", i, " evaluation time : ", end_time-start_time, " sec")
    torch.save(model, exp_dir+"/model_" + str(i) + ".pt")


def run_inference(model: nn.Module,
                     dataloader: DataLoader, 
                     device: torch.cuda.device):
    
    model.eval()
    full_y = torch.Tensor([]).to(device)
    full_logits = torch.Tensor([]).to(device)
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X).squeeze()

            full_y = torch.cat([full_y, y])
            full_logits = torch.cat([full_logits, logits])

    return full_y, full_logits


for i in range(0, epoch_num + finetune_epoch_num):
    models_paths = [exp_dir+"/model_" + str(i) + ".pt"]
    pcam_model = torch.load(models_paths[0], weights_only = False)
    pcam_model = pcam_model.to(device)

    # First create tta_num augmented dataloaders
    tta_num = 1
    logits = []
    for j in range(0, tta_num):
        test_set_augment = PcamDatasetKaggle(root=dataset_dir, split="test", transform = deepcopy(transform_data_augment)) #For TTA
        test_dataloader_augment = DataLoader(test_set_augment, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6, persistent_workers = True)
        for modelp in models_paths:
            pcam_model = torch.load(modelp, weights_only = False)
            pcam_model = pcam_model.to(device)
            test_y, test_logits = run_inference(pcam_model, test_dataloader, device)
            logits.append(test_logits)
            test_y_augm, test_logits_aum = run_inference(pcam_model, test_dataloader_augment, device)
            logits.append(test_logits_aum)
        
    # Average logits
    logits_stacked = torch.stack(logits)
    mean_logits = torch.mean(logits_stacked, dim = 0, keepdims=True)

    #Create submission file with final predictions
    image_ids = [img.replace('.tif', '') for img in test_set.imgs.tolist()]
    test_preds = torch.sigmoid(mean_logits)

    submission_df = pd.DataFrame({
        'id': image_ids,
        'label': test_preds.squeeze().detach().cpu().numpy()
    })

    submission_df.to_csv(exp_dir+'/submission_'+str(i)+'.csv', index=False)


sub_path = exp_dir + '/submission_4.csv'
model_path = models_paths[0]
#you need to update your creds at the top for this
#!kaggle competitions submit -c histopathologic-cancer-detection -f {sub_path} -m {model_path}


i = 4
models_paths = [exp_dir+"/model_" + str(i) + ".pt"]
pcam_model = torch.load(models_paths[0], weights_only = False)
pcam_model = pcam_model.to(device)
test_y, test_logits = run_inference(pcam_model, val_dataloader, device)
test_y_augment, test_logits_augment = run_inference(pcam_model, val_dataloader_augment, device)
full_y = torch.cat([test_y, test_y_augment])
full_logits = torch.cat([test_logits, test_logits_augment])


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(full_y.detach().cpu().numpy(), torch.sigmoid(full_logits).detach().cpu().numpy())
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.grid(alpha=0.3)
plt.show()



# Find best threshold index (maximize TPR-FPR).
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]



best_threshold




