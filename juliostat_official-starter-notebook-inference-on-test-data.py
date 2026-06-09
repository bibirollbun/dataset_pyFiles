import csv
import numpy as np 
import pandas as pd
import timm 
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import logging
import time
import os

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()])
_logger = logging.getLogger('inference')


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
        

class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, image_path 


df_species_ids = pd.read_csv('/kaggle/input/plantclef-2025/species_ids.csv')

df_metadata = pd.read_csv('/kaggle/input/plantclef-2025/PlantCLEF2024_single_plant_training_metadata.csv', sep=';', dtype={'partner': str})
id_to_species = df_metadata[['species_id', 'species']].drop_duplicates().set_index('species_id')

df_metadata.head()


device = torch.device('cuda')
model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                          pretrained=False,
                          num_classes=len(df_species_ids),
                          checkpoint_path='/kaggle/input/dinov2_patch14_reg4_onlyclassifier_then_all/pytorch/default/3/model_best.pth.tar')
model = model.to(device)
model = model.eval()


data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)


batch_size = 32
top_k = 15
min_score = 0.01 



class_map = df_species_ids['species_id'].to_dict()
dataset = TestDataset(image_folder='/kaggle/input/plantclef-2025/PlantCLEF2025_test_images/PlantCLEF2025_test_images/',
                      transform=timm.data.create_transform(**data_config, is_training=False))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

image_predictions = {}

# Initialize batch time tracking
batch_time = AverageMeter()
end = time.time()

with torch.no_grad():
    for batch_idx, (images, image_paths) in enumerate(dataloader):
        images = images.to(device)
        outputs = model(images)  # Perform inference on the batch
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get the top-k values and their indices
        values, indices = torch.topk(probabilities, top_k, dim=1)
        
        # Filter based on the probability threshold
        values_np = values.cpu().numpy()
        indices_np = indices.cpu().numpy()
        
        for i in range(values_np.shape[0]):
            # Filtered class indices above the threshold
            filtered_indices = indices_np[i][values_np[i] >= min_score]
            
            # Convert class indices to class labels
            filtered_labels = [class_map.get(idx, 'Unknown') for idx in filtered_indices]

            # Get the image name without the extension
            image_name = os.path.splitext(os.path.basename(image_paths[i]))[0]

            image_predictions[image_name] = filtered_labels
        
        batch_time.update(time.time() - end)
        end = time.time()

        # Log info at specified frequency
        if batch_idx % 10 == 0:  # You can set your log frequency here
            _logger.info(f'Predict: [{batch_idx}/{len(dataloader)}] '
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})')  


df_run = pd.DataFrame(list(image_predictions.items()), columns=['quadrat_id', 'species_ids'])
df_run['species_ids'] = df_run['species_ids'].apply(str)
df_run.to_csv("submission.csv", sep=',', index=False, quoting=csv.QUOTE_ALL)

