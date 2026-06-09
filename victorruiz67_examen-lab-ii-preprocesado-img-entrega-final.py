!pip install -q transformers


import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import cv2

import torch
from transformers import CLIPProcessor, CLIPModel
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings('ignore')

print("✅ Librerías importadas")


# Configuración
TRAIN_IMG_PATH = '../input/petfinder-adoption-prediction/train_images'
TEST_IMG_PATH = '../input/petfinder-adoption-prediction/test_images'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")

print(f"Train: {len(train)}, Test: {len(test)}")


def get_pet_images(img_path, pet_ids):
    pet_images = defaultdict(list)
    all_images = set(os.listdir(img_path))
    for pet_id in pet_ids:
        pet_imgs = sorted([f for f in all_images if f.startswith(pet_id)])
        pet_images[pet_id] = pet_imgs
    return pet_images

train_pet_images = get_pet_images(TRAIN_IMG_PATH, train.index.tolist())
test_pet_images = get_pet_images(TEST_IMG_PATH, test.index.tolist())


def calculate_blur(img_array):
    """Calcula el blur score usando Laplacian variance."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_image_metadata(img_path, pet_images):
    features = []
    for pet_id, images in tqdm(pet_images.items(), desc="Metadatos"):
        feat = {
            'img_count': len(images),
            'img_has_photo': 1 if len(images) > 0 else 0,
        }
        
        if len(images) > 0:
            try:
                img = Image.open(os.path.join(img_path, images[0]))
                img_array = np.array(img.convert('RGB'))
                
                feat['img_width'] = img.size[0]
                feat['img_height'] = img.size[1]
                feat['img_aspect_ratio'] = img.size[0] / max(1, img.size[1])
                feat['img_size'] = img.size[0] * img.size[1]
                feat['img_mean_brightness'] = img_array.mean()
                feat['img_std_brightness'] = img_array.std()
                
                # Blur score (técnica de ganadores)
                feat['img_blur_score'] = calculate_blur(img_array)
                
                # Color dominance
                feat['img_red_mean'] = img_array[:,:,0].mean()
                feat['img_green_mean'] = img_array[:,:,1].mean()
                feat['img_blue_mean'] = img_array[:,:,2].mean()
            except:
                for k in ['img_width', 'img_height', 'img_aspect_ratio', 'img_size', 
                          'img_mean_brightness', 'img_std_brightness', 'img_blur_score',
                          'img_red_mean', 'img_green_mean', 'img_blue_mean']:
                    feat[k] = 0
        else:
            for k in ['img_width', 'img_height', 'img_aspect_ratio', 'img_size', 
                      'img_mean_brightness', 'img_std_brightness', 'img_blur_score',
                      'img_red_mean', 'img_green_mean', 'img_blue_mean']:
                feat[k] = 0
        
        features.append(feat)
    
    df = pd.DataFrame(features)
    df.index = list(pet_images.keys())
    return df

train_meta = extract_image_metadata(TRAIN_IMG_PATH, train_pet_images)
test_meta = extract_image_metadata(TEST_IMG_PATH, test_pet_images)
print(f"Metadatos: {train_meta.shape[1]} features")


# Cargar DenseNet121 
densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
print(f"DenseNet121 cargado. Output: {densenet.output_shape}")


def extract_densenet_features(img_path, pet_images, model, img_size=(224, 224)):
    all_features_1st = []  # Primera imagen
    all_features_2nd = []  # Segunda imagen (si existe)
    pet_ids = []
    output_dim = model.output_shape[-1]  # 1024
    
    for pet_id, images in tqdm(pet_images.items(), desc="DenseNet"):
        # Primera imagen
        if len(images) >= 1:
            try:
                img = load_img(os.path.join(img_path, images[0]), target_size=img_size)
                img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
                feat_1 = model.predict(img_array, verbose=0).flatten()
            except:
                feat_1 = np.zeros(output_dim)
        else:
            feat_1 = np.zeros(output_dim)
        
        # Segunda imagen 
        if len(images) >= 2:
            try:
                img = load_img(os.path.join(img_path, images[1]), target_size=img_size)
                img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
                feat_2 = model.predict(img_array, verbose=0).flatten()
            except:
                feat_2 = np.zeros(output_dim)
        else:
            feat_2 = np.zeros(output_dim)
        
        all_features_1st.append(feat_1)
        all_features_2nd.append(feat_2)
        pet_ids.append(pet_id)
    
    return np.array(all_features_1st), np.array(all_features_2nd), pet_ids

print("Extrayendo DenseNet features...")
train_dn_1st, train_dn_2nd, train_ids = extract_densenet_features(TRAIN_IMG_PATH, train_pet_images, densenet)
test_dn_1st, test_dn_2nd, test_ids = extract_densenet_features(TEST_IMG_PATH, test_pet_images, densenet)

print(f"DenseNet 1st shape: {train_dn_1st.shape}")
print(f"DenseNet 2nd shape: {train_dn_2nd.shape}")


# SVD sobre DenseNet features
N_SVD = 32

# SVD para primera imagen
svd_1st = TruncatedSVD(n_components=N_SVD, random_state=42)
train_dn1_svd = svd_1st.fit_transform(train_dn_1st)
test_dn1_svd = svd_1st.transform(test_dn_1st)

# SVD para segunda imagen
svd_2nd = TruncatedSVD(n_components=N_SVD, random_state=42)
train_dn2_svd = svd_2nd.fit_transform(train_dn_2nd)
test_dn2_svd = svd_2nd.transform(test_dn_2nd)

train_dn1_df = pd.DataFrame(train_dn1_svd, columns=[f'dn1_svd_{i}' for i in range(N_SVD)], index=train_ids)
test_dn1_df = pd.DataFrame(test_dn1_svd, columns=[f'dn1_svd_{i}' for i in range(N_SVD)], index=test_ids)

train_dn2_df = pd.DataFrame(train_dn2_svd, columns=[f'dn2_svd_{i}' for i in range(N_SVD)], index=train_ids)
test_dn2_df = pd.DataFrame(test_dn2_svd, columns=[f'dn2_svd_{i}' for i in range(N_SVD)], index=test_ids)

print(f"DenseNet SVD: {N_SVD * 2} features totales")


# CLIP para features semánticas
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("CLIP cargado")


def extract_clip_features(img_path, pet_images, model, processor, device):
    all_features = []
    pet_ids = []
    
    for pet_id, images in tqdm(pet_images.items(), desc="CLIP"):
        if len(images) == 0:
            all_features.append(np.zeros(512))
            pet_ids.append(pet_id)
            continue
        
        # Solo primera imagen para CLIP
        try:
            img = Image.open(os.path.join(img_path, images[0])).convert('RGB')
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy().flatten())
        except:
            all_features.append(np.zeros(512))
        
        pet_ids.append(pet_id)
    
    return np.array(all_features), pet_ids

train_clip, train_clip_ids = extract_clip_features(TRAIN_IMG_PATH, train_pet_images, clip_model, clip_processor, DEVICE)
test_clip, test_clip_ids = extract_clip_features(TEST_IMG_PATH, test_pet_images, clip_model, clip_processor, DEVICE)

print(f"CLIP shape: {train_clip.shape}")


# SVD sobre CLIP
N_CLIP_SVD = 32

svd_clip = TruncatedSVD(n_components=N_CLIP_SVD, random_state=42)
train_clip_svd = svd_clip.fit_transform(train_clip)
test_clip_svd = svd_clip.transform(test_clip)

train_clip_df = pd.DataFrame(train_clip_svd, columns=[f'clip_svd_{i}' for i in range(N_CLIP_SVD)], index=train_clip_ids)
test_clip_df = pd.DataFrame(test_clip_svd, columns=[f'clip_svd_{i}' for i in range(N_CLIP_SVD)], index=test_clip_ids)

print(f"CLIP SVD: {N_CLIP_SVD} features")


# Combinar todo
train_features = train_meta.join(train_dn1_df).join(train_dn2_df).join(train_clip_df)
test_features = test_meta.join(test_dn1_df).join(test_dn2_df).join(test_clip_df)

# Reindexar
train_features = train_features.reindex(train.index).fillna(0)
test_features = test_features.reindex(test.index).fillna(0)

print(f"\n✅ Total features de imagen: {train_features.shape[1]}")


# Guardar
train_features.to_parquet("train.parquet")
test_features.to_parquet("test.parquet")

print(f"Guardado: train.parquet {train_features.shape}")
print(f"Guardado: test.parquet {test_features.shape}")







