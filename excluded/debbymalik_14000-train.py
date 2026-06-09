!pip install --no-index --no-deps /kaggle/input/lavis-pretrained/salesforce-lavis/transformers* 
!pip install --no-index --no-deps /kaggle/input/lavis-pretrained/salesforce-lavis/hugging*


pip install -U transformers


!pip install --upgrade transformers


import os
import gc
import copy
import time
import random
import joblib

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Utils
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# For Transformer Models
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration
from torch.optim import AdamW

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "False"


CONFIG = {
    "seed": 2023,
    "epochs": 10,
    "model_name": "Salesforce/blip-image-captioning-base",
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "learning_rate": 5e-7,  
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 5e-7,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "competition": "SD", 
}

# Load processor from model name
CONFIG["processor"] = AutoProcessor.from_pretrained(CONFIG["model_name"])


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])


import pandas as pd
from PIL import Image
import os

# Fungsi untuk memuat gambar dan menambahkan ke dataset
def load_image_and_add_to_dataset(csv_file, image_folder):
    # Membaca file CSV
    df = pd.read_csv(csv_file)
    
    # List untuk menyimpan data baru
    data_list = []
    
    # Iterasi melalui DataFrame dan muat gambar
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, row['image'])
        image = Image.open(image_path)
        data = {
            'image': image,
            'caption': row['caption']
        }
        data_list.append(data)
    
    return data_list

# Path ke file CSV dan folder gambar
csv_file = "/kaggle/input/14000datas/blipdatasetptbxlengcap_14000.csv"
image_folder = "/kaggle/input/ptb-xl-ecg-processed/PTB-XL ECG Dataset/PTB-XL ECG Dataset"

# Muat gambar dan tambahkan ke dataset
Dataset = load_image_and_add_to_dataset(csv_file, image_folder)

# Mengakses elemen pertama
print(Dataset[0])

# Mengakses informasi gambar dan prompt dari elemen pertama
image = Dataset[0]['image']
caption = Dataset[0]['caption']
print("Caption:", caption)
print("Image:", image)


Dataset[6]


Dataset[2]["caption"]


Dataset[2]["image"]


from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp (val + test)
train_df, temp_df = train_test_split(Dataset, test_size=0.2, random_state=42)

# Second split: split temp_df equally into 10% val and 10% test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


print(len(train_df))
print(len(val_df))
print(len(test_df))


import pandas as pd
import os

# Membuat DataFrame baru untuk disimpan sebagai CSV
csv_data_list = []

for item in test_df:
    # Menggunakan path gambar atau nama file untuk disimpan di CSV
    image_path = os.path.basename(item['image'].filename)
    caption = item['caption']
    csv_data_list.append({'caption': caption, 'image': image_path})

# Membuat DataFrame
csv_df = pd.DataFrame(csv_data_list)

# Menyimpan DataFrame ke file CSV
csv_file_path = 'Test_Dataset_BLIP-ECG-14000.csv'
csv_df.to_csv(csv_file_path, index=False)

print(f"Test_Dataset_BLIP-ECG telah disimpan di {csv_file_path}")


from torch.utils.data import Dataset

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["caption"], 
                                  padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


train_dataset = ImageCaptioningDataset(train_df, CONFIG['processor'])
valid_dataset = ImageCaptioningDataset(val_df, CONFIG['processor'])


train_dataset[0].keys()


model = BlipForConditionalGeneration.from_pretrained(CONFIG['model_name'])


import gc
import torch
from tqdm import tqdm

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data['input_ids'].to(device)
        pixel_values = data['pixel_values'].to(device)
        
        batch_size = input_ids.size(0)

        outputs = model(input_ids=input_ids, 
                        pixel_values=pixel_values, 
                        labels=input_ids)
                
        loss = outputs.loss
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_loss


import gc
import torch
from tqdm import tqdm

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        input_ids = data['input_ids'].to(device)
        pixel_values = data['pixel_values'].to(device)
        
        batch_size = input_ids.size(0)

        outputs = model(input_ids=input_ids, 
                        pixel_values=pixel_values, 
                        labels=input_ids)
                
        loss = outputs.loss
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)  # Removed LR since no optimizer here
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_loss


import time
import copy
import numpy as np
from collections import defaultdict

def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        train_epoch_loss = train_one_epoch(
            model, optimizer, scheduler, 
            dataloader=train_loader, 
            device=device, 
            epoch=epoch
        )
        
        val_epoch_loss = valid_one_epoch(
            model, valid_loader, 
            device=device, 
            epoch=epoch
        )
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        print(f"Epoch {epoch} -> Train Loss: {train_epoch_loss:.4f} | Valid Loss: {val_epoch_loss:.4f}")
        
        # Save the best model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss:.4f} ---> {val_epoch_loss:.4f})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "BestLoss.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Model saved to {PATH}{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 60)
    ))
    print(f"Best Loss: {best_epoch_loss:.4f}")
    
    # load best model weights before returning
    model.load_state_dict(best_model_wts)
    
    return model, history


def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


from torch.utils.data import DataLoader
from torch.optim import AdamW
import os

# Create Dataloaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG['train_batch_size'])
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=CONFIG['valid_batch_size'])

model.to(CONFIG['device'])

# Define Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)

model, history = run_training(
    model, optimizer, scheduler,
    device=CONFIG['device'],
    num_epochs=CONFIG['epochs'],
    train_loader=train_loader,
    valid_loader=valid_loader
)

# Save the model in Kaggle working directory
save_path = 'blip-ecg-base-10epoch-14000.pth'
torch.save(model.state_dict(), save_path)
print(f"Training finished and model saved as {save_path}")

# List files in working directory to confirm save
print("Files in working directory:", os.listdir('.'))


import matplotlib.pyplot as plt

plt.plot(range(1, CONFIG['epochs'] + 1), history['Train Loss'], label='Train Loss')
plt.plot(range(1, CONFIG['epochs'] + 1), history['Valid Loss'], label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


import torch
import os

# Path baru untuk menyimpan model
model_output_path = '/kaggle/working/blip-ecg-capt-14000-10epoch.pth'

# Simpan model
torch.save(model.state_dict(), model_output_path)
print(f"Model berhasil disimpan di {model_output_path}")

# Cek isi direktori kerja untuk konfirmasi
print("Isi direktori kerja saat ini:")
for file in os.listdir('/kaggle/working'):
    print("-", file)


#

