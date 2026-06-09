import numpy as np 
import pandas as pd
import timm 
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import time
import os
import torchvision.transforms as T
from torch.amp import autocast
from matplotlib import pyplot as plt
from kornia import tensor_to_image
from kornia.contrib import extract_tensor_patches, compute_padding
import csv


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class PatchDataset(Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches.squeeze(0)
        self.transform = transform

    def __len__(self):
        return self.patches.size(0)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        if self.transform:
            patch = self.transform(patch)
        return patch


class TestDataset(Dataset):
    def __init__(self, image_folder, patch_size=518, stride=259, transform=None, use_pad=False):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.transform = transform
        self.use_pad = use_pad
        self.patch_size = patch_size
        self.stride = stride
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image).unsqueeze(0)
        
        h, w = image.shape[-2:]
        
        if self.use_pad:
            pad = compute_padding(original_size=(h, w), window_size=self.patch_size, stride=self.stride)
            patches = extract_tensor_patches(image, self.patch_size, self.stride, padding=pad)
        else:
            patches = extract_tensor_patches(image, self.patch_size, self.stride)

        return patches, image_path


df_species_ids = pd.read_csv('/kaggle/input/plantclef-2025/species_ids.csv')

df_metadata = pd.read_csv('/kaggle/input/plantclef-2025/PlantCLEF2024_single_plant_training_metadata.csv', sep=';', dtype={'partner': str})
class_map = df_species_ids['species_id'].to_dict() # dictionary to map the species model Id with the species Id

df_metadata.head()


device = torch.device('cuda')
model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                          pretrained=False,
                          num_classes=len(df_species_ids),
                          checkpoint_path='/kaggle/input/dinov2_patch14_reg4_onlyclassifier_then_all/pytorch/default/3/model_best.pth.tar')
model = model.to(device)
model = model.eval()


data_config = timm.data.resolve_model_data_config(model)
model_input_size, model_mean, model_std = data_config['input_size'][1], data_config['mean'], data_config['std']


batch_size = 64
min_score = 0.1
top_k_tile = 3
patch_size = model_input_size
stride = int(model_input_size / 2)
use_pad = True


image_to_tensor = T.ToTensor()


dataset = TestDataset(image_folder='/kaggle/input/plantclef-2025/PlantCLEF2025_test_images/PlantCLEF2025_test_images/',
                      patch_size=patch_size,
                      stride=stride,
                      use_pad=True,
                      transform=image_to_tensor)
dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

image_predictions = {}
pred_result = {}

# Initialize batch time tracking
batch_time = AverageMeter()
end = time.time()

with torch.no_grad():
    for batch_idx, (patches, image_path) in enumerate(dataloader):
        image_results = {}
        quadrat_id = os.path.splitext(os.path.basename(image_path[0]))[0]
        transform_patch = T.Normalize(mean=model_mean, std=model_std)
        patch_dataset = PatchDataset(patches[0], transform=transform_patch)
        patch_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=False)

        sp_id_list = []
        sp_id_probs = []
        for batch_patches in patch_loader:
            batch_patches = batch_patches.to(device)
            
            with autocast('cuda'):
                outputs = model(batch_patches)  # Perform inference on the batch
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, 10)
                mask = top_probs > min_score
                filtered_top_probs = top_probs[mask]
                k = min(top_k_tile, filtered_top_probs.numel())
                _, indices = torch.topk(top_probs.view(-1), k)
                plant_sp_ids = list(top_indices.view(-1)[indices].cpu().numpy())
                sp_prob = list(top_probs.view(-1)[indices].cpu().numpy())
                sp_ids = [class_map[int(idx)] for idx in plant_sp_ids]
                sp_id_prob = [v for v in zip(sp_ids, sp_prob)]
                sp_id_probs.extend(sp_id_prob)
                sp_id_list.extend(sp_ids)
                
                
        sp_id_list = list(set(sp_id_list))
        # store the prediction
        image_predictions[quadrat_id] = sp_id_list
        pred_result[quadrat_id] = sp_id_probs
        batch_time.update(time.time() - end)
        end = time.time()

        # Log info at specified frequency
        if batch_idx % 10 == 0:  # You can set your log frequency here
            print(f'Predict: [{batch_idx}/{len(dataloader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})')  


df_run = pd.DataFrame(list(image_predictions.items()), columns=['quadrat_id', 'species_ids'])
df_run['species_ids'] = df_run['species_ids'].apply(str)
df_run.to_csv("submission.csv", sep=',', index=False, quoting=csv.QUOTE_ALL)










