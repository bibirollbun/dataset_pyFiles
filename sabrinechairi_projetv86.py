import pandas as pd
import re
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizer
import numpy as np
from transformers import DistilBertModel
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.nn as nn


# Lire le fichier CSV en ignorant les lignes mal formatées
file_path = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/train.csv'

# Nettoyage léger en filtrant les lignes avec un nombre de colonnes incorrect
with open(file_path) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

# Afficher les colonnes disponibles
print("Columns in the dataset:")
print(df.columns)

# Afficher le nombre de lignes et de colonnes
print(f"Total number of rows in the dataset: {df.shape[0]}")
print(f"Total number of columns in the dataset: {df.shape[1]}")

# Afficher un aperçu des 5 premières lignes
print("Dataset preview:")
print(df.head())


# Prétraiter les étiquettes
def preprocess_labels(labels):
    return [int(label) for label in labels.split()]

df['Labels'] = df['Labels'].apply(preprocess_labels)
print(df['Labels'].head())


mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(df['Labels'])
print("Classes détectées par MultiLabelBinarizer:", mlb.classes_)


df['BinaryLabels'] = list(binary_labels)
print("Dataframe with binarized labels:\n", df.head())


# Charger le tokenizer de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Analyser la longueur maximale des captions pour définir MAX_SEQ_LENGTH
caption_lengths = df['Caption'].apply(lambda x: len(x.split()))
MAX_SEQ_LENGTH = int(np.percentile(caption_lengths, 100))
#MAX_SEQ_LENGTH = int(np.percentile(caption_lengths, 95))  # Utiliser le 95e percentile
#MAX_SEQ_LENGTH = min(512, MAX_SEQ_LENGTH)  # DistilBERT a une limite de 512 tokens
print(f"Longueur séquence max retenue : {MAX_SEQ_LENGTH}")





# Fonction pour tokenizer une caption et retourner les IDs et les tokens
def tokenize_caption(caption):
    tokenized = tokenizer.encode_plus(
        caption,
        add_special_tokens=True,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = tokenized['input_ids'].squeeze().tolist()  # Convertir en liste
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Convertir les IDs en tokens
    return input_ids, tokens

# Appliquer la fonction de tokenization à toutes les captions
df[['Tokenized IDs', 'Tokenized Tokens']] = df['Caption'].apply(
    lambda x: pd.Series(tokenize_caption(x))
)

# Afficher les premières lignes du DataFrame avec les nouvelles colonnes
print(df.head())


# Diviser les données en ensembles d'entraînement et de test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train dataset size: {len(train_df)}")
print(f"Test dataset size: {len(test_df)}")


# Déplacer le modèle sur le GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Charger le modèle DistilBERT
txt_model  = DistilBertModel.from_pretrained('distilbert-base-uncased')
# Déplacer le modèle sur le GPU si disponible
txt_model = txt_model.to(device)


# Convertir les Tokenized IDs et attention_mask en tenseurs et les transfert vers device
train_input_ids = torch.tensor(train_df['Tokenized IDs'].tolist()).to(device)
train_attention_mask = (train_input_ids != 0).long().to(device)

test_input_ids = torch.tensor(test_df['Tokenized IDs'].tolist()).to(device)
test_attention_mask = (test_input_ids != 0).long().to(device)



# Fonction pour extraire les features textuelles
def extract_text_features(input_ids, attention_mask, txt_model , device, batch_size=32):
    txt_model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_input_ids = input_ids[i:i + batch_size].to(device)
            batch_attention_mask = attention_mask[i:i + batch_size].to(device)
            outputs = txt_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            batch_features = outputs.last_hidden_state[:, 0, :]  # Utiliser le token [CLS]
            features.append(batch_features.cpu())
    return torch.cat(features, dim=0)

# Extraire les features textuelles pour l'ensemble d'entraînement et de test
train_text_features = extract_text_features(train_input_ids, train_attention_mask, txt_model , device)
test_text_features = extract_text_features(test_input_ids, test_attention_mask, txt_model , device)

# Afficher les dimensions des features
print(f"Shape of train text features: {train_text_features.shape}")
print(f"Shape of test text features: {test_text_features.shape}")


# Charger le modèle ResNet50 pour les images
#img_model = models.resnet50(pretrained=True)
img_model = models.convnext_tiny(pretrained=True)
#img_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
# Supprimer la dernière couche fully connected
img_model = torch.nn.Sequential(*list(img_model.children())[:-1])

# Mettre le modèle en mode évaluation
img_model.eval()

# Déplacer le modèle sur le GPU si disponible
img_model = img_model.to(device)






transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Charger l'image
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])  # Colonne 0 : ImageID
        image = Image.open(img_name).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        # Récupérer les input_ids (tokenization du texte)
        input_ids = torch.tensor(self.dataframe.iloc[idx]['Tokenized IDs'], dtype=torch.long)
        
        # Récupérer les BinaryLabels
        binary_labels = torch.tensor(self.dataframe.iloc[idx]['BinaryLabels'], dtype=torch.float32)
        
        return image, input_ids, binary_labels


# Répertoire des images
image_dir = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/data'

# Créer les datasets
train_image_dataset = ImageDataset(train_df, image_dir, transform=transform)
test_image_dataset = ImageDataset(test_df, image_dir, transform=transform)

# Créer les DataLoaders
batch_size = 32
train_image_loader = DataLoader(train_image_dataset, batch_size=batch_size, shuffle=False)
test_image_loader = DataLoader(test_image_dataset, batch_size=batch_size, shuffle=False)


# Afficher un échantillon du dataset
image, input_ids, binary_labels = train_image_dataset[0]
print("Image shape:", image.shape)  # Doit être [3, 224, 224]
print("Input IDs:", input_ids)  # Doit être un tenseur de longueur MAX_SEQ_LENGTH
print("Binary Labels:", binary_labels)  # Doit être un tenseur de longueur égale au nombre de classes


def extract_image_features(img_model, dataloader, device):
    img_model.eval()
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)  # Extraire les images du tuple
            features_batch = img_model(images)
            features_batch = features_batch.view(features_batch.size(0), -1)  # Aplatir les features
            features.append(features_batch.cpu())
    
    return torch.cat(features, dim=0)

# Extraire les features pour l'ensemble d'entraînement et de test
train_image_features = extract_image_features(img_model, train_image_loader, device)
test_image_features = extract_image_features(img_model, test_image_loader, device)

# Afficher les dimensions des features
print(f"Shape of train image features: {train_image_features.shape}")
print(f"Shape of testing image features: {test_image_features.shape}")


# Fusionner les embeddings textuels et image
#train_multimodal_features = torch.cat([train_text_features, train_image_features], dim=1)
#test_multimodal_features = torch.cat([test_text_features, test_image_features], dim=1)
# Réduire la dimension des features visuelles
#fc_reduce = nn.Linear(2048, 768).to(device)
#train_image_features_reduced = fc_reduce(train_image_features.to(device))
#test_image_features_reduced = fc_reduce(test_image_features.to(device))

# Déplacer les features textuelles sur le même appareil que les features visuelles réduites
train_text_features = train_text_features.to(device)
test_text_features = test_text_features.to(device)

# Combiner les features
train_multimodal_features = torch.cat([train_text_features, train_image_features.to(device)], dim=1)
test_multimodal_features = torch.cat([test_text_features, test_image_features.to(device)], dim=1)

# Afficher les dimensions
print(f"Shape of train multimodal features: {train_multimodal_features.shape}")
print(f"Shape of test multimodal features: {test_multimodal_features.shape}")


class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultimodalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Couche fully connected 1
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Couche fully connected 2
        self.relu = nn.ReLU()  # Fonction d'activation
        self.dropout = nn.Dropout(0.5)  # Dropout pour éviter le surajustement

    def forward(self, x):
        if len(x.shape) > 2:  # Si la forme est incorrecte, aplatissez-la
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

 
# Définir les dimensions
input_dim = train_multimodal_features.shape[1]  
hidden_dim = 512  # Dimension de la couche cachée
# Nombre de classes détectées
num_classes = len(mlb.classes_)
output_dim = num_classes  # Nombre de classes 

# Initialiser le modèle
model = MultimodalClassifier(input_dim, hidden_dim, output_dim).to(device)


# Définir la fonction de perte
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss avec logits

# Définir l'optimiseur
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Modifier l'optimiseur avec weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-5)


from torch.utils.data import Dataset, DataLoader, TensorDataset
# Conversion des labels en tenseurs
train_labels = torch.tensor(train_df['BinaryLabels'].tolist(), dtype=torch.float32)
test_labels = torch.tensor(test_df['BinaryLabels'].tolist(), dtype=torch.float32)

# Créer les DataLoaders
train_dataset = TensorDataset(train_multimodal_features, train_labels)
test_dataset = TensorDataset(test_multimodal_features, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



from sklearn.metrics import f1_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calcul du F1-Score
    f1 = f1_score(np.array(all_labels).ravel(), np.array(all_preds).ravel(), average='binary')  # 'binary' pour une classification binaire
    return f1


# Initialiser des listes pour stocker les pertes et le F1-Score
train_losses = []
val_losses = []
val_f1_scores = []

# Early Stopping
best_val_loss = float('inf')
patience = 5
counter = 0

# Boucle d'entraînement
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for batch_features, batch_labels in train_loader:
       
        if len(batch_features.shape) > 2:  # Si la forme est incorrecte, aplatissez-la
            batch_features = batch_features.view(batch_features.size(0), -1)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            if len(batch_features.shape) > 2:  # Si la forme est incorrecte, aplatissez-la
                batch_features = batch_features.view(batch_features.size(0), -1)
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            epoch_val_loss += loss.item()

    epoch_val_loss /= len(test_loader)
    val_losses.append(epoch_val_loss)

    val_f1 = evaluate_model(model, test_loader, device)
    val_f1_scores.append(val_f1)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val F1-Score: {val_f1:.4f}")


import matplotlib.pyplot as plt

# Tracer les courbes de perte et de F1-Score
plt.figure(figsize=(12, 5))

# Tracer la perte d'entraînement et de validation
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()

# Tracer le F1-Score de validation
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, label="Validation F1-Score")
plt.xlabel("Epochs")
plt.ylabel("F1-Score")
plt.title("Validation F1-Score")
plt.legend()

plt.show()


# Charger le modèle sauvegardé
multi_modal_model = MultimodalClassifier(input_dim, hidden_dim, output_dim).to(device)
multi_modal_model.load_state_dict(torch.load("./best_model.pth"))
multi_modal_model.eval()

# Évaluer le modèle sur l'ensemble de test
val_loss = 0.0
correct_predictions = 0
num_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = multi_modal_model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        correct_predictions += (predictions == labels).float().sum().item()
        num_samples += labels.numel()

val_loss /= len(test_loader)
val_accuracy = correct_predictions / num_samples

print("ERREUR TEST AVEC MEILLEUR MODEL")
print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')



import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Charger le fichier test.csv
file_path = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/test.csv'

# Charger les données de test
with open(file_path) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    test_df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

# Afficher les colonnes disponibles
print("Columns in the dataset:")
print(test_df.columns)

# Afficher un aperçu des données textuelles et des étiquettes
print("\nText data preview:")
print(test_df[['Caption', 'ImageID']].head())

# Vérifier les données manquantes dans les colonnes utilisées
print("\nMissing values in the selected columns:")
print(test_df[['Caption', 'ImageID']].isnull().sum())

# Nombre total de lignes dans le jeu de données
print(f"\nTotal number of rows in the dataset: {test_df.shape[0]}")


# Appliquer la fonction de tokenization à toutes les captions
test_df[['Tokenized IDs', 'Tokenized Tokens']] = test_df['Caption'].apply(
    lambda x: pd.Series(tokenize_caption(x))
)

# Afficher les premières lignes du DataFrame avec les nouvelles colonnes
print(test_df.head())


class ImageDatasetTest(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Charger l'image
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])  # Colonne 0 : ImageID
        image = Image.open(img_name).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        # Récupérer les input_ids (tokenization du texte)
        input_ids = torch.tensor(self.dataframe.iloc[idx]['Tokenized IDs'], dtype=torch.long)
        
        return image, input_ids


class ImageDatasetTest(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Charger l'image
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])  # Colonne 0 : ImageID
        image = Image.open(img_name).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        # Récupérer les input_ids (tokenization du texte)
        input_ids = torch.tensor(self.dataframe.iloc[idx]['Tokenized IDs'], dtype=torch.long)
        
        return image, input_ids


# Répertoire des images de test
test_image_dir = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/data'

# Créer le dataset pour les images de test
test_image_dataset = ImageDatasetTest(test_df, test_image_dir, transform=transform)

# Créer le DataLoader pour les images de test
test_image_loader = DataLoader(test_image_dataset, batch_size=batch_size, shuffle=False)

# Extraire les features visuelles pour les données de test
test_image_features = extract_image_features(img_model, test_image_loader, device)

# Tokenizer les captions de test
test_input_ids = torch.tensor(test_df['Tokenized IDs'].tolist())
test_attention_mask = (test_input_ids != 0).long()

# Extraire les features textuelles pour les données de test
test_text_features = extract_text_features(test_input_ids, test_attention_mask, txt_model, device)

# Fusionner les features textuelles et visuelles pour les données de test
test_multimodal_features = torch.cat([test_text_features.to(device), test_image_features.to(device)], dim=1)

# Afficher les dimensions des features multimodales de test
print(f"Shape of test multimodal features: {test_multimodal_features.shape}")

# Charger le modèle sauvegardé
multi_modal_model = MultimodalClassifier(input_dim, hidden_dim, output_dim).to(device)
multi_modal_model.load_state_dict(torch.load("./best_model.pth"))
multi_modal_model.eval()

# Prédire les labels pour l'ensemble de test
all_preds = []
with torch.no_grad():
    for inputs in DataLoader(TensorDataset(test_multimodal_features), batch_size=32, shuffle=False):
        inputs = inputs[0].to(device)
        outputs = multi_modal_model(inputs)
        preds = torch.sigmoid(outputs) > 0.5  # Seuil de classification
        all_preds.append(preds.cpu())

# Convertir les prédictions binaires en labels originaux
all_preds = torch.cat(all_preds, dim=0).numpy()
predicted_labels = mlb.inverse_transform(all_preds)  # Convertir les binaires en labels

# Créer le fichier de soumission
submission = pd.DataFrame({
    'ImageID': test_df['ImageID'],
    'Labels': [' '.join(map(str, labels)) for labels in predicted_labels]
})

# Sauvegarder le fichier de soumission
submission_file_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")
print("Submission file preview:")
print(submission.head(10))

