import sys
import warnings
warnings.filterwarnings('ignore')

# Instalar transformers compatible con Python 3.7
!{sys.executable} -m pip install -q --disable-pip-version-check transformers==4.29.2 2>/dev/null

# Reiniciar kernel
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# Importación de Librerías
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Cargar modelo vit-large

model_name = "google/vit-large-patch16-224"

print(f"Loading {model_name}...")
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded successfully!")


# Funciones para extraer features

def extract_features_batch(image_paths, batch_size=32):
    all_features = []
    image_names = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing"):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(path).convert('RGB') for path in batch_paths]
        
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.pooler_output
        
        all_features.append(features.cpu().numpy())
        image_names.extend([os.path.basename(p) for p in batch_paths])
    
    return np.vstack(all_features), image_names

def extract_features_all_images(image_dir):
    from collections import defaultdict
    
    all_pics = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    full_paths = [f'{image_dir}/{f}' for f in all_pics]
    print(f"Found {len(all_pics)} total images")
    
    all_features, image_names = extract_features_batch(full_paths)
    
    pet_features = defaultdict(list)
    for feat, img_name in zip(all_features, image_names):
        pet_id = img_name.replace('.jpg', '').split('-')[0]
        pet_features[pet_id].append(feat)
    
    aggregated_features = {}
    for pet_id, features_list in pet_features.items():
        aggregated_features[pet_id] = np.mean(features_list, axis=0)
    
    pet_ids = list(aggregated_features.keys())
    features_array = np.array([aggregated_features[pid] for pid in pet_ids])
    
    df = pd.DataFrame(
        features_array,
        index=pet_ids,
        columns=[f"img_feat_{i + 1}" for i in range(features_array.shape[1])]
    )
    
    print(f"Aggregated to {len(df)} unique pets")
    return df


# Imagenes train
train_img = extract_features_all_images('../input/petfinder-adoption-prediction/train_images')
train_img.to_parquet("train.parquet")


# Imagenes test
test_img = extract_features_all_images('../input/petfinder-adoption-prediction/test_images')
test_img.to_parquet("test.parquet")


print(f"Train shape: {train_img.shape}, Test shape: {test_img.shape}")

