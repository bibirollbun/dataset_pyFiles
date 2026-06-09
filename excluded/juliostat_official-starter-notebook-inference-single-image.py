import numpy as np
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
import timm 
import torch


df_species_ids = pd.read_csv('/kaggle/input/plantclef-2025/species_ids.csv')
df_metadata = pd.read_csv('/kaggle/input/plantclef-2025/PlantCLEF2024_single_plant_training_metadata.csv', sep=';', dtype={'partner': str})
id_to_species = df_metadata[['species_id', 'species']].drop_duplicates().set_index('species_id')


img = Image.open('/kaggle/input/plantclef-2025/PlantCLEF2025_test_images/PlantCLEF2025_test_images/GUARDEN-CBNMed-30-4-16-3-20240428.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()


img.size


device = torch.device('cuda')
model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                          pretrained=False,
                          num_classes=len(df_species_ids),
                          checkpoint_path='/kaggle/input/dinov2_patch14_reg4_onlyclassifier_then_all/pytorch/default/3/model_best.pth.tar')
model = model.to(device)
model = model.eval()


# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

with torch.no_grad():
    if img != None:
        img = transforms(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)  # unsqueeze single image into batch of 1
        top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1), k=5)
        top5_probabilities = top5_probabilities.cpu().detach().numpy()
        top5_class_indices = top5_class_indices.cpu().detach().numpy()
    
        for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
            species_id = df_species_ids.iloc[cid].item()
            species = id_to_species.loc[species_id].item()
            print(species_id, species, proba)


