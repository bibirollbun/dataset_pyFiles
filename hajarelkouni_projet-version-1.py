# Data process
import pandas as pd
import numpy as np

# Graphing
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Image loading
from PIL import Image
import glob
from numpy import asarray

# Output
from sys import stdout
import csv
import copy
from random import seed
from random import randint
import random
import time
import datetime
import zipfile
#from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, Module


import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from skimage.transform import rotate
from skimage.util import random_noise



#read csv file
data_train= pd.read_csv('/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/train.csv', on_bad_lines='skip')
data_train.head(5)



#read csv file
data_test= pd.read_csv('/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/test.csv', on_bad_lines='skip')
data_test.head(5)



import pandas as pd

def process_and_clean_csv(filepath, expected_fields=3):
    """
    Nettoie et charge un fichier CSV dans un DataFrame.

    Args:
    - filepath (str): Chemin vers le fichier d'entrée.
    - expected_fields (int): Nombre de champs attendus par ligne (par défaut 3).

    Returns:
    - pd.DataFrame: Le DataFrame nettoyé.
    """
    cleaned_lines = []
    invalid_lines = []

    # Étape 1 : Lire le fichier brut pour identifier les lignes problématiques
    with open(filepath, 'r') as file:
        for idx, line in enumerate(file):
            # Vérifier si le nombre de virgules correspond au nombre attendu
            if line.count(',') == (expected_fields - 1):  # (fields - 1) virgules pour N champs
                cleaned_lines.append(line)
            else:
                invalid_lines.append((idx + 1, line))  # Stocker les lignes problématiques

    # Afficher un résumé des lignes invalides
    print(f"Nombre de lignes problématiques ignorées : {len(invalid_lines)}")
    if invalid_lines:
        print("Exemples de lignes problématiques :")
        for i, line in invalid_lines[:5]:  # Afficher les 5 premières lignes problématiques
            print(f"Ligne {i} : {line.strip()}")

    # Étape 2 : Charger les lignes nettoyées directement dans un DataFrame
    from io import StringIO  # Utilisé pour lire les données en mémoire
    cleaned_data = "\n".join(cleaned_lines)  # Combiner les lignes nettoyées en un seul texte
    try:
        data = pd.read_csv(StringIO(cleaned_data))
    except Exception as e:
        print(f"Erreur lors de la lecture des données nettoyées : {e}")
        return None

    # Étape 3 : Nettoyer les colonnes (si applicable)
    if 'Labels' in data.columns:
        # Nettoyer 'Labels' : supprimer les espaces inutiles
        data['Labels'] = data['Labels'].str.strip().str.replace(r'\s+', ' ', regex=True)

    if 'Caption' in data.columns:
        # Nettoyer 'Caption' : supprimer les caractères spéciaux
        data['Caption'] = data['Caption'].str.replace(r'[^\w\s]', '', regex=True)

  
    # Retourner le DataFrame nettoyé
    return data



# Chemin du fichier d'entraînement
train_dataset_path = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/train.csv'

# Nettoyer et charger les données dans un DataFrame
data_train = process_and_clean_csv(train_dataset_path, expected_fields=3)

# Afficher un aperçu des données nettoyées
if data_train is not None:
    print(data_train.head())



# Chemin du fichier de test
test_dataset_path = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/test.csv'

# Nettoyer et charger les données de test
data_test = process_and_clean_csv(test_dataset_path, expected_fields=2)

# Afficher un aperçu des données nettoyées
if data_test is not None:
    print(data_test.head())



import os
from tqdm import tqdm
from PIL import Image
import numpy as np

def load_images_from_folder(folder_path):
    
    """
    Charge les images depuis un dossier, les convertit en RGB, les redimensionne à 64x64 et les stocke dans un dictionnaire.

    Args:
    - folder_path (str): Chemin vers le dossier contenant les images.

    Returns:
    - dict: Un dictionnaire où les clés sont les noms des fichiers et les valeurs sont les tableaux numpy représentant les images.
    """
    print("Loading images from folder...")
    image_dict = {}
    
    # Vérification que le dossier existe
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return image_dict
    
    # Liste des fichiers dans le dossier
    file_list = os.listdir(folder_path)
    
    image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # Inclure plusieurs formats
    
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        try:
            image_path = os.path.join(folder_path, image_file)
           
            if os.path.isfile(image_path):  # Vérifier si c'est bien un fichier
                with Image.open(image_path) as img:
                     
                    # Convertir l'image en RGB
                    img_rgb = img.convert('RGB')
                    # Redimensionner l'image à 64x64
                    img_resized = img_rgb.resize((64, 64))
                    # Convertir l'image redimensionnée en tableau NumPy
                    image_array = np.asarray(img_resized)
                    # Stocker l'image dans le dictionnaire
                    image_dict[image_file] = image_array
                   
        except Exception as e:
            print(f"Erreur lors du chargement de {image_file}: {e}")
    
    print(f"\nNombre total d'images chargées: {len(image_dict)}")
    return image_dict

# Exemple d'utilisation
folder_path = '/kaggle/input/multi-label-classification-competition-2023/COMP5329S1A2Dataset/data'
images = load_images_from_folder(folder_path)
for image_name in images:
    image_path = os.path.join(folder_path, image_name)
  

# Aperçu des clés du dictionnaire chargé
print("Quelques images chargées:", list(images.keys())[:5])



from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Resize((28, 28)),       # Redimensionner à 28x28
    transforms.Grayscale(num_output_channels=1),  # Convertir en niveaux de gris
    transforms.ToTensor(),            # Convertir en tenseur
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation
])

data_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


all_single_labels= []
all_single_labels += [int(x) for lst in [eachlab.split(" ") for eachlab in data_train.Labels] for x in lst] 
unique_labels = np.unique(all_single_labels)
print(unique_labels)
label_to_index = {n: i for i, n in enumerate(unique_labels)}
index_to_label = {i: n for i, n in enumerate(unique_labels)}
output_dim = len(label_to_index)
print("Number of unique labels :",output_dim)



import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('ggplot')
plt.hist(all_single_labels,bins=unique_labels)
plt.title("Distribution of labels")
plt.ylabel('Count')
plt.xlabel('Labels')
plt.xticks(unique_labels)
plt.show()


seed=27

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


from torch.utils import data
from PIL import Image
import torch

class ImageData(data.Dataset):
    def __init__(self, df, transform, image_dict, test=False):
        """
        Args:
        - df (DataFrame): DataFrame contenant les noms des fichiers et leurs labels.
        - transform (callable): Transformation à appliquer aux images.
        - image_dict (dict): Dictionnaire contenant les données d'images.
        - test (bool): Si True, le dataset est en mode test (pas de labels).
        """
        self.df = df
        self.test = test
        self.transform = transform
        self.image_dict = image_dict  # Stocke le dictionnaire d'images
        self.image_arr = self.df.iloc[:, 0]  # Noms des fichiers image
        if not self.test:  # Charger les labels si non test
            self.label_df = self.df.iloc[:, 1]
        self.data_len = len(self.df.index)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # Obtenir le nom de l'image et charger les données correspondantes
        image_name = self.image_arr[idx]
        image_data = self.image_dict[image_name]  # Récupérer les données depuis le dictionnaire
        img_tensor = self.transform(Image.fromarray(image_data))  # Appliquer les transformations

        if not self.test:  # Retourner l'image et les labels pour l'entraînement
            image_labels = self.label_df[idx]
            label_tensor = torch.zeros((1, output_dim))
            for label in image_labels.split():
                label_tensor[0, label_to_index[int(label)]] = 1
            return img_tensor, label_tensor.squeeze()

        return img_tensor



import nltk
nltk.download('punkt')

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

caption_train = [tknzr.tokenize(s) for s in data_train.Caption]
caption_test  = [tknzr.tokenize(s) for s in data_test.Caption]


nltk.download('stopwords')
from nltk.corpus import stopwords as sw
stop_words = sw.words('english')  #Only English Stopwords


text_train_ns=[]
for tokens in caption_train:
  filtered_sentence = [w for w in tokens if not w in stop_words]
  text_train_ns.append(filtered_sentence)

text_test_ns=[]
for tokens in caption_test:
    filtered_sentence =  [w for w in tokens if not w in stop_words] 
    text_test_ns.append(filtered_sentence)


import copy
import datetime

#Adding all words into a list called word_sequence
word_sequence = []
for corpus_temp in [text_train_ns,text_test_ns]:
  for sentence in corpus_temp:
    word_sequence.extend(sentence)


print("Total unique words:",len(set(word_sequence)))


word_list = list(set(word_sequence))
#print(len(word_list))
word_list.sort()

# make dictionary so that we can reference each index of unique word during training
word_dict = {w: i for i, w in enumerate(word_list)}


seq_length = 8 #Derived from above analysis

def add_padding(oldcorpus, seq_length):
    corpus = copy.deepcopy(oldcorpus) #Performing a  deepcopy so that text_train_le, text_test_le does not get effected
    output = []
    for sentence in corpus:
      if len(sentence)>seq_length:
        output.append(sentence[:seq_length])  #Truncating the sentence if it is longer than seq_length
      else:
        for j in range(seq_length-len(sentence)): #Padding the sentence if it is smaller than seq_length
          sentence.append("<PAD>")
        output.append(sentence)
    return output

text_train_pad = add_padding(text_train_ns,seq_length )
text_test_pad = add_padding(text_test_ns,seq_length )


train_tags = []

for eachlab in data_train.Labels:
  label_tensor = [0]*output_dim
  for label in eachlab.split():
    label_tensor[label_to_index[int(label)]] = 1  #converting lavel to index of label
  train_tags.append(label_tensor)

train_tags = np.array(train_tags)


from sklearn.model_selection import train_test_split

# Définir la taille du jeu de validation
VALIDATION_SET_SIZE = 0.2  # 20% pour la validation

# Effectuer la séparation entre train et validation
train_df, val_df = train_test_split(data_train, test_size=VALIDATION_SET_SIZE)

# Réinitialiser les index des DataFrames
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Afficher la taille des ensembles
print(f"Validation_Data Length: {len(val_df)}\nTrain_Data Length: {len(train_df)}")


# Train dataset
train_dataset = ImageData(train_df, data_transform, images)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

# Validation dataset
val_dataset = ImageData(val_df, data_test_transforms, images)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True)

# Full dataset - to be used during feature extractions
full_dataset = ImageData(train_df, data_test_transforms, images)
full_loader = DataLoader(dataset=full_dataset, batch_size=128, shuffle=False)

# Test dataset
test_dataset = ImageData(data_test, data_test_transforms, images, test=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)



print(f"Taille du dataset : {len(train_dataset)}")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_df


 label_to_index 


class Digits_NN(Module):
    def __init__(self):
        super(Digits_NN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8,
                                     kernel_size=(3,3), stride=(1,1))

        self.conv2 = torch.nn.Conv2d(8, 16,
                                     kernel_size=(3,3), stride=(1,1))

        self.conv3 = torch.nn.Conv2d(16, 32,
                                     kernel_size=(3,3), stride=(1,1))
        self.pool = torch.nn.MaxPool2d( (2,2) )

        self.classifier = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = torch.relu(x)

        #############################"
        bs = x.shape[0]
        x = x.reshape(bs, -1)

        x = self.classifier(x)
        #x = torch.softmax(x, -1)
        #J'ai désactivé le softmax parceque CrossEntropyLoss applique elle même le softmax. 
        #Pour que CrossEntropyLoss fonctionne on doit donc lui fournir les valeurs sans softmax
  
        return x
model = Digits_NN()
model.to(device)

# Récupérer les deux éléments spécifiques
x1 = train_dataset[10][0]  # Premier élément
x2 = train_dataset[11][0]  # Deuxième élément

# Empiler les deux exemples dans un seul batch
x_batch = torch.stack([x1, x2])  # Empile les deux exemples dans un batch de taille 2

# Déplacer les données sur le bon appareil (GPU/CPU)
x_batch = x_batch.to(device)

# Passer le batch dans le modèle
x_batch = model(x_batch)


x_batch


def eval_test(loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():        
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward: le modèle prédit les labels du batch
            outputs = model(inputs)
            labels = labels.long()
            labels = torch.nn.functional.one_hot(labels, num_classes=10)
            labels = labels[:, 0, :]
       
        
            labels = labels.float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)
    

PATIENCE = 5
current_patience = 0
BEST_ERROR = 1e9
num_epochs = 50  # Nombre d'epochs (nombre de parcours de la base de données d'apprentissage)

model = Digits_NN()
model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss() #
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_dataset):
        # Torch cumule les gradients par défaut, il faut reinitialiser les gradients à chaque batch pour éviter de réutiliser les gradients précédents
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward: le modèle prédit les labels du batch
        outputs = model(inputs)
        
        # On calcule l'erreur
        labels = labels.long()
        labels = torch.nn.functional.one_hot(labels, num_classes=18)
   
    
        labels = labels.float()

        loss = criterion(outputs, labels)
      
        
        # Backward propagation: on propage l'erreur dans le réseau de neurones
        loss.backward()
        
        # Mise à jour des poids grâce au SGD
        optimizer.step()

        total_loss += loss.item()
        
    total_loss = total_loss/ len(train_loader)
    test_total_loss = eval_test(test_loader)

    if test_total_loss < BEST_ERROR:
        BEST_ERROR = test_total_loss
        current_patience = 0
        torch.save(model, "./model.h5")
    else:
        current_patience += 1

    if current_patience>PATIENCE:
        break
        
    print(epoch, total_loss, test_total_loss)

print("ERREUR TEST AVEC MEILLEUR MODEL")
model = torch.load("./model.h5")
model.eval()
print( eval_test(train_loader) )

