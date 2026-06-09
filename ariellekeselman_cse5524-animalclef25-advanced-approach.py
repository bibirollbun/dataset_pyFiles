import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import cv2
import numpy as np


df = pd.read_csv('../input/animal-clef-2025/metadata.csv')
df.tail()



# Define your data directories (Update with actual paths)
data_dirs = {    
    "SeaTurtlesD": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/database/turtles-data/data/images/t001",
    "SeaTurtlesQ": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/query/images",
    "LynxsD": "/kaggle/input/animal-clef-2025/images/LynxID2025/database",
    "LynxsQ": "/kaggle/input/animal-clef-2025/images/LynxID2025/query",
    "SalamandersD": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/database/images",
    "SalamandersQ": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/query/images"
}

# Check if directories exist
for label, path in data_dirs.items():
    if not os.path.exists(path):
        print(f"⚠ Warning: Directory {path} does not exist for {label}.")
    else:
        print(f"✅ Directory found: {path} ({label})")

def show_sample_images():
    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Add more formats if needed

    for label, dir_path in data_dirs.items():
        if not os.path.exists(dir_path):
            print(f"⚠ Skipping {label}: Directory does not exist.")
            continue

        # Load images
        sample_images = []
        for root, _, files in os.walk(dir_path):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if img_path.lower().endswith(valid_formats):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (150, 150))
                        sample_images.append(img)
                    if len(sample_images) == 5:  # Stop after 5 images
                        break
            if len(sample_images) == 5:
                break

        # Plot images
        if sample_images:
            plt.figure(figsize=(10, 10))
            for i, img in enumerate(sample_images):
                plt.subplot(1, 5, i + 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"{label}")
            plt.show()
        else:
            print(f"⚠ No valid images found in {dir_path} for {label}.")

# Call the function to display images
show_sample_images()



# Set random seed for reproducibility
tf.random.set_seed(42)

IMG_SIZE=150


# Function to load images
def load_images():
    X = []  # Image data
    Y = []  # Labels

    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Supported formats

    for label, dir_path in data_dirs.items():
        if not os.path.exists(dir_path):
            print(f"⚠ Warning: Directory {dir_path} does not exist for {label}. Skipping...")
            continue

        # Read images from the directory
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            if img_path.lower().endswith(valid_formats):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (150, 150))  # Resize for consistency
                    X.append(img)
                    Y.append(label)  # Store label
    
    return X, Y

# Load the dataset
X, Y = load_images()

# Convert to numpy arrays
if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
    raise ValueError(f"Mismatch in dataset sizes: X={len(X)}, Y={len(Y)}")

X = np.array(X, dtype='float32') / 255.0  # Normalize images
Y = np.array([list(data_dirs.keys()).index(y) for y in Y], dtype='int32')  # Convert labels to indices

# Shuffle and split dataset
X, Y = shuffle(X, Y, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"✅ Dataset successfully loaded: Train={len(X_train)}, Test={len(X_test)}")



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.neighbors import NearestNeighbors
import random
from PIL import Image



#  Variable setup 
IMAGE_SIZE = IMG_SIZE
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20
EMBEDDING_SIZE = 512
NEW_THRESHOLD = 0.6  # Tunable threshold for 'new_individual'
K_NEIGHBORS=5



# Directories based on existing data_dirs
TRAIN_DIR = {
    "SeaTurtle": data_dirs["SeaTurtlesD"],
    "Salamander": data_dirs["SalamandersD"],
    "Lynx": data_dirs["LynxsD"]
}
TEST_DIR = {
    "SeaTurtle": data_dirs["SeaTurtlesQ"],
    "Salamander": data_dirs["SalamandersQ"],
    "Lynx": data_dirs["LynxsQ"]
}
SPECIES_LIST = list(TRAIN_DIR.keys())


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Loads each species’ dataset and merges them into one large training set to train a single model on multiple species
def create_combined_species_dataset(species_list, train_dir, transform):
    datasets = []
    for species in species_list:
        ds = AdvancedAnimalDataset(train_dir[species], species_label=species, transform=transform)
        datasets.append(ds)
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    return combined_dataset



#  Retrieve images for training dataset and save paths
class AdvancedAnimalDataset(Dataset):
    def __init__(self, root_dir, species_label, transform=None):
        self.root_dir = root_dir  
        self.species_label = species_label  
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # Assign numerical label to each species
        label_map = {"SeaTurtle": 0, "Salamander": 1, "Lynx": 2}
        self.label = label_map[species_label]  # Integer label for the folder

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        for img_file in os.listdir(self.root_dir):
            if img_file.lower().endswith(valid_exts):
                img_path = os.path.join(self.root_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(self.label)

        print(f"[INFO] Loaded {len(self.image_paths)} images from {self.species_label}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label, self.species_label



#  Retrieve images for test dataset and save paths
class AdvancedTestDataset(Dataset):
    def __init__(self, root_dir, species, transform=None):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            self.root_dir = os.path.join(root_dir, f'{species}ID2022' if 'SeaTurtle' in species else f'{species}ID2025', 'query') 
        self.transform = transform
        self.image_paths = []
        self.image_ids = []
        self.species = species
        self.label_map = {"SeaTurtle": 0, "Salamander": 1, "Lynx": 2}
        self._load_data()

    def _load_data(self):
        for filename in os.listdir(self.root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(self.root_dir, filename))
                self.image_ids.append(filename.split('.')[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_id = self.image_ids[idx]
        class_label = self.label_map[self.species]  # new line
        if self.transform:
            image = self.transform(image)
        return image, image_id, self.species, class_label


# Re-identification model class
class ReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, weights=None): 
        super(ReIDModel, self).__init__()

        # Load ResNet with or without weights
        resnet = models.resnet50(weights=weights)

        # Remove the classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Add embedding and classification heads
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        embedding = self.fc(x)
        embedding = self.bn(embedding)
        out = self.classifier(embedding)
        return embedding, out


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def train_reid_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    triplet_batches = 0

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings, _ = model(images)

        # Group embeddings by label
        anchor_list, positive_list, negative_list = [], [], []

        for i in range(len(labels)):
            anchor = embeddings[i]
            label = labels[i]

            # Find indices of positive and negative samples
            positive_indices = (labels == label).nonzero(as_tuple=False).squeeze(1)
            negative_indices = (labels != label).nonzero(as_tuple=False).squeeze(1)

            # Skip if there are no valid positives/negatives
            if len(positive_indices) <= 1 or len(negative_indices) == 0:
                continue

            # Choose one positive (not the anchor itself)
            pos_index = positive_indices[positive_indices != i]
            if len(pos_index) == 0:
                continue
            pos_index = pos_index[0]

            # Choose hardest negative (closest to anchor)
            neg_embeds = embeddings[negative_indices]
            anchor_dist = torch.norm(anchor - neg_embeds, dim=1)
            neg_index = negative_indices[anchor_dist.argmin()]

            # Append triplet
            anchor_list.append(anchor)
            positive_list.append(embeddings[pos_index])
            negative_list.append(embeddings[neg_index])

        if len(anchor_list) == 0:
            continue  # Skip batch if no valid triplets

        anchor_tensor   = torch.stack(anchor_list)
        positive_tensor = torch.stack(positive_list)
        negative_tensor = torch.stack(negative_list)

        # Compute triplet loss
        loss = criterion(anchor_tensor, positive_tensor, negative_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        triplet_batches += 1

        if triplet_batches == 0:
            print(" No valid triplets found in this epoch.")
            return torch.tensor(0.0)
        avg_loss = total_loss / triplet_batches
        print(f'Train Loss: {avg_loss:.4f}')
        return avg_loss



#  Extract features from training data
def extract_features(model, data_loader, device):
    model.eval()
    all_features = []
    all_individuals = []
    with torch.no_grad():
        for images, labels, individuals in data_loader:
            images = images.to(device)
            embeddings, _ = model(images)
            all_features.append(embeddings.cpu().numpy())
            all_individuals.extend(individuals)
    return np.concatenate(all_features, axis=0), np.array(all_individuals)


from collections import Counter
def train_model():
     set_seed()
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print(f"Using device: {device}")

     train_transform = transforms.Compose([
         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

     print('--- Training on All Species ---')
     train_dataset = create_combined_species_dataset(SPECIES_LIST, TRAIN_DIR, train_transform)
     combined_labels = []
     for _, label, _ in train_dataset:
         combined_labels.append(label)
     print("Combined label distribution:", Counter(combined_labels))
     
     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
     
     # Standard training setup
     num_classes = 3
     model = ReIDModel(num_classes, embedding_size=EMBEDDING_SIZE, weights=None).to(device)
     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
     criterion = TripletLoss(margin=0.3)
   
     for epoch in range(EPOCHS):
         print(f'Epoch {epoch+1}/{EPOCHS}')
         train_reid_model(model, train_loader, optimizer, criterion, device)
     # --- After training the model on all species ---
     # Extract features from the combined training set
     features, individuals = extract_features(model, train_loader, device)
     # Save the unified trained model and features
     torch.save(model.state_dict(), "reid_model_all_species.pth")
     torch.save(features, "train_features.pth")
     torch.save(individuals, "train_individuals.pth")
     
     print("Training complete and model saved.")



# Prediction Approach
def predict_advanced(model, test_loader, train_features, train_individuals, device, new_threshold=NEW_THRESHOLD, k_neighbors=5):
    model.eval()
    predictions = []
    image_ids = []
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    knn.fit(train_features)

    with torch.no_grad():
        for images, img_ids, species, _ in test_loader:
            images = images.to(device)
            embeddings, _ = model(images)
            for embedding in embeddings.cpu().numpy():
                distances, indices = knn.kneighbors(embedding.reshape(1, -1))
                avg_distance = np.mean(distances)

                if avg_distance > new_threshold:
                    predictions.append('new_individual')
                else:
                    nearest_individuals = train_individuals[indices[0]]
                    unique_individuals, counts = np.unique(nearest_individuals, return_counts=True)
                    predicted_individual_base = unique_individuals[np.argmax(counts)]

                    year = '2022' if 'SeaTurtle' in species else '2025'
                    predictions.append(f'{species}ID{year}_{predicted_individual_base}')
            image_ids.extend(list(img_ids))
    return predictions, image_ids


def predict_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Inference on device: {device}")

    label_inv = {0: "SeaTurtle", 1: "Salamander", 2: "Lynx"}        
    species_to_num = {v: k for k, v in label_inv.items()}          
    
    #  Load trained model & saved features 
    model = ReIDModel(num_classes=3,
                      embedding_size=EMBEDDING_SIZE,
                      weights=None).to(device)
    model.load_state_dict(torch.load("reid_model_all_species.pth", map_location=device, weights_only=True))    
    model.eval()

    train_features   = torch.load("train_features.pth")
    train_individual = torch.load("train_individuals.pth")
    train_features = torch.tensor(train_features)  # convert to torch tensor
    train_features = F.normalize(train_features, dim=1).cpu().numpy()


    # Build query dataset 
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    query_ds   = []
    for sp in ["SeaTurtle", "Salamander", "Lynx"]:
        query_ds.extend( AdvancedTestDataset(TEST_DIR[sp], sp, transform=test_transform) )

    query_loader = DataLoader(query_ds,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    #  Predict with thresholding 
    preds, img_ids = predict_advanced(model,
                                      query_loader,
                                      train_features,
                                      np.array(train_individual),
                                      device,
                                      new_threshold=NEW_THRESHOLD,
                                      k_neighbors=K_NEIGHBORS)

    # Optional accuracy at species level 
    # build true label list from dataset items
    true_species = [species_to_num[item[2]] for item in query_ds]  # item = (img, id, species)
    pred_species = []
    for p in preds:
        if "SeaTurtle" in p:   pred_species.append(0)
        elif "Salamander" in p:pred_species.append(1)
        elif "Lynx" in p:      pred_species.append(2)
        else:                  pred_species.append(-1)   # new_individual

    valid = [i for i, ps in enumerate(pred_species) if ps != -1]
    if valid:
        acc = accuracy_score([true_species[i] for i in valid],
                             [pred_species[i] for i in valid])
        print(f" Species‑level top‑1 accuracy on known individuals: {acc:.4f}")
    else:
        print(" No known‑individual predictions; species accuracy not computed.")

    # Save CSV 
    out_df = pd.DataFrame({"image_id": img_ids,
                           "identity" : preds})
    out_df.to_csv("animal_predictions.csv", index=False)
    print("Saved predictions to  /kaggle/working/animal_predictions.csv")

    #  Preview first 10 rows -----
    print("\nFirst 10 rows:")
    print(out_df.head(10))



train_model()


predict_model()

