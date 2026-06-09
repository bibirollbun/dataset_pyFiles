import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# ===========================
# CONFIGURACIÓN INICIAL
# ===========================
class Config:
    # Hiperparámetros ajustables
    SEED = 8
    IMG_SIZE = 64
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.01
    WEIGHT_DECAY = 1e-6
    TRAIN_RATIO = 0.9  # Para dividir train/val internamente
    
    # Rutas
    DATASET_PATH = "/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes"
    TRAIN_CSV = os.path.join(DATASET_PATH, "train_dataset.csv")
    TEST_CSV = os.path.join(DATASET_PATH, "test_dataset.csv")
    
    # Configuración del modelo
    NUM_CLASSES = 10

# Fijar semillas para reproducibilidad
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# ===========================
# PREPARACIÓN DE DATOS
# ===========================
class FruitDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test = is_test # determina si el dataset es de pruebas o de entrenamiento
        
        # Mapeo de clases a índices
        # Si el dataset no es de prueba, se extraen las clases unicas, se ordenan y se crea un diccionario para mapear cada clase a un indice numerico
        if not is_test:
            self.classes = sorted(self.data['Label'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(Config.DATASET_PATH, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image
        else:
            label = self.class_to_idx[self.data.iloc[idx, 1]]
            return image, label

# Transformaciones con aumento de datos
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar y dividir datos
full_train_dataset = FruitDataset(Config.TRAIN_CSV, transform=train_transform)
train_size = int(Config.TRAIN_RATIO * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(Config.SEED)
)

# Actualizar transformaciones para validación
val_dataset.dataset.transform = val_transform

test_dataset = FruitDataset(Config.TEST_CSV, transform=val_transform, is_test=True)

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=True, num_workers=4, pin_memory=True)
# Se usa un batch mayor para mejorar la eficiencia en el calculo de metricas
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE*2,
                       num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE*2,
                        num_workers=4, pin_memory=True)



# ==============
# ARQUITECTURA
# ==============
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = FruitCNN(Config.NUM_CLASSES).to(device)




# ===========================
# FUNCIONES DE ENTRENAMIENTO
# ===========================
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=0):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        if scheduler:
            scheduler.step()

        # Calcular metricas como floats
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (running_corrects.double() / len(train_dataset)).cpu().item()
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Fase de validación
        val_loss, val_acc = evaluate_model(model, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Best val Acc: {best_acc:.4f}')
    return history

def evaluate_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = (running_corrects.double() / len(val_dataset)).cpu().item()
    return epoch_loss, epoch_acc #floats



# ===========================
# ENTRENAMIENTO CON VALIDACIÓN
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.01)

history = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=Config.EPOCHS
)




# ===========================
# VISUALIZACIÓN DE RESULTADOS
# ===========================
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)



# ===========================
# GENERACIÓN DE SUBMISIÓN
# ===========================
def generate_submission(model, test_loader, class_names, device, filename='submission.csv'):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    
    # Mapear índices a nombres de clase sin el " 1"
    class_names_clean = [cls.split(' 1')[0] for cls in full_train_dataset.classes]
    submission = pd.DataFrame({
        'ID': range(len(all_preds)),
        'Outcome': [class_names_clean[p] for p in all_preds]
    })
    
    submission.to_csv(filename, index=False)
    print(f'Submission file {filename} created!')

# Generar archivo final
generate_submission(
    model=model,
    test_loader=test_loader,
    class_names=full_train_dataset.classes,
    device=device
)


# Verificar si el archivo existe
if os.path.exists("submission.csv"):
    print("✅ submission.csv está en:", os.getcwd())
else:
    print("❌ submission.csv no se generó.")

