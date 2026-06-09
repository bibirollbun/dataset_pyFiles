import cv2
import numpy as np
import pandas as pd
from skimage import feature, filters, measure
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# déclarer le dossier de base
base_path='/kaggle/input/fsdm-classification-2025-2026-fruit-vision/Fruit_dataset_2 classes/Fruit_dataset_2 classes/'


import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def display_random_images(train_path, num_images=4):
    """Afficher des images aléatoires du dataset d'entraînement"""
    # Obtenir tous les chemins d'images
    all_images = []
    class_names = ['apple', 'malay_apple']
    
    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append((os.path.join(class_path, img_file), class_name))
    
    # Sélectionner aléatoirement
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Afficher les images
    fig, axes = plt.subplots(2, 2, figsize=(15, 4))
    axes = axes.ravel()
    
    for idx, (img_path, class_name) in enumerate(selected_images):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'{class_name}\n{os.path.basename(img_path)}', fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            print(f"Erreur avec l'image {img_path}: {e}")
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
# Utilisation
display_random_images(base_path+'/train', num_images=4)


def extract_color_features(image):
    ...
    return features

def extract_texture_features(image):
    ...
    return features

def extract_shape_features(image):
    ...
    return features


def extract_all_features(image_path):
    """Extract all features from an image"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        # Resize image for consistency (optional)
        # image = cv2.resize(image, (256, 256))
        
        all_features = []
        
        # Extract different types of features
        all_features.extend(extract_color_features(image))
        all_features.extend(extract_texture_features(image))
        all_features.extend(extract_shape_features(image))
        
        return all_features
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def load_dataset_and_extract_features(base_path, dataset_type='train'):
    """Load dataset and extract features"""
    features_list = []
    labels_list = []
    
    class_mapping = {'apple': 0, 'malay_apple': 1}
    
    for class_name, class_label in class_mapping.items():
        class_path = os.path.join(base_path, dataset_type, class_name)

        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} does not exist")
            continue
            
        print(f"Processing {class_name} images...")
        
        for image_file in tqdm(os.listdir(class_path)):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_path, image_file)
                #print(image_path)
                features = extract_all_features(image_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(class_label)
    
    return np.array(features_list), np.array(labels_list)


# Extract features pour training
print("Extracting training features...")
X_train, y_train = load_dataset_and_extract_features(base_path, 'train')

print(f"Training features shape: {X_train.shape}")


# Extract features pour validation
print("Extracting validation features...")
X_val, y_val = load_dataset_and_extract_features(base_path, 'val')

print(f"Validation features shape: {X_val.shape}")


# A completer




# A completer


# Create submission file
submission = pd.DataFrame({
    'id': test_image_names,
    'class': test_predictions,
    'probability_apple': test_probabilities[:, 0],
    'probability_malay_apple': test_probabilities[:, 1]
})

# Map labels back to class names
submission['class'] = submission['class'].map({0: 'apple', 1: 'malay_apple'})

# Save submission
submission[['id', 'class']].to_csv('/kaggle/working/submission_clss_fruit.csv', index=False)
print("Fichier sauvegardé dans /kaggle/working/submission_clss_fruit.csv")

