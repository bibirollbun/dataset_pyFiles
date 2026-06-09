# Celda de c贸digo para participantes
print('Nombre_Completo_1')
print('Nombre_Completo_2')


# Celda de c贸digo
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.metrics import balanced_accuracy_score # Para evaluar
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader





# Celda de c贸digo
# (Ajuste clave)
# Esta clase de Dataset est谩 adaptada para PyTorch.
# Nota: nn.CrossEntropyLoss espera 铆ndices de clase (0, 1, 2...)
# no vectores one-hot. Por eso devolvemos el label directamente.

class OrganCMNIST(Dataset):
    def __init__(self, x, y=None, transform=None):
        self._x = x
        self._y = y.squeeze() if y is not None else None
        self._transform = transform
        self._num_classes = 11

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, idx):
        # Las im谩genes son 28x28, las convertimos a 1x28x28 (canal, alto, ancho)
        image = self._x[idx].reshape(1, 28, 28)

        # Convertimos a torch.Tensor de tipo float
        image = torch.tensor(image, dtype=torch.float32)

        if self._transform is not None:
            image = self._transform(image)

        if self._y is None:
            return image

        # Devolvemos el 铆ndice de la clase como un LongTensor
        label = torch.tensor(self._y[idx], dtype=torch.long)
        return image, label


# Celda de c贸digo
# Funci贸n de normalizaci贸n (隆importante para las redes neuronales!)
def normalize(tensor_image):
    # Normaliza entre 0 y 1 (ya que los datos originales van de 0 a 255)
    return tensor_image / 255.0


import os
os.listdir("/kaggle/input/hackathon-de-ia-unsch")


data_train = np.load('/kaggle/input/hackathon-de-ia-unsch/train_set.npz', allow_pickle=True)
data_val = np.load('/kaggle/input/hackathon-de-ia-unsch/val_set.npz',allow_pickle=True)
data_test = np.load('/kaggle/input/hackathon-de-ia-unsch/test_set.npz',allow_pickle=True)


# Extraer im谩genes y etiquetas
x_train, y_train = data_train['images'], data_train['labels']
x_val, y_val     = data_val['images'], data_val['labels']
x_test           = data_test['images']  # test no tiene labels



# --- Crear Datasets ---
train_dataset = OrganCMNIST(x_train, y_train, transform=normalize)
val_dataset   = OrganCMNIST(x_val, y_val, transform=normalize)
test_dataset  = OrganCMNIST(x_test, transform=normalize)

# --- Crear DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")



class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        # Un bloque residual est谩ndar: Lineal -> BatchNorm -> SiLU -> Lineal -> BatchNorm
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.SiLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # La "magia" de ResNet: sumamos la entrada original al resultado del bloque
        return self.activation(x + self.block(x))

class Net(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()

        # --- ARQUITECTURA TIPO RESNET (MED-AI OPTIMIZED) ---
        self.input_layer = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )

        # A帽adimos bloques residuales (capas profundas que no degradan el gradiente)
        self.res_block1 = ResidualBlock(512)
        self.res_block2 = ResidualBlock(512)

        self.intermediate = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        
        self.res_block3 = ResidualBlock(256)

        self.output_layer = nn.Linear(256, num_classes)
        # --------------------------------------------------

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.intermediate(x)
        x = self.res_block3(x)
        logits = self.output_layer(x)
        return logits


# Celda de c贸digo
def train_epoch(model, loader, optimizer, criterion):
    model.train() # Pone el modelo en modo entrenamiento
    running_loss = 0.0

    for images, labels in loader:
        # 1. Poner gradientes a cero
        optimizer.zero_grad()

        # 2. Forward pass (obtener predicciones)
        outputs = model(images)

        # 3. Calcular la p茅rdida
        loss = criterion(outputs, labels)

        # 4. Backward pass (calcular gradientes)
        loss.backward()

        # 5. Actualizar pesos
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate_epoch(model, loader, criterion):
    model.eval() # Pone el modelo en modo evaluaci贸n
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Desactiva el c谩lculo de gradientes
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Guardar predicciones para la m茅trica
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # Usamos la m茅trica de la competici贸n: Balanced Accuracy
    acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, acc


# Celda de c贸digo
def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs):
    # Log (historial) para graficar
    log_dict = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("Iniciando entrenamiento...")
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Balanced Acc: {val_acc:.4f}")

        log_dict['epoch'].append(epoch)
        log_dict['train_loss'].append(train_loss)
        log_dict['val_loss'].append(val_loss)
        log_dict['val_acc'].append(val_acc)

    print("Entrenamiento finalizado.")
    return log_dict

# Funci贸n para graficar la p茅rdida
def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], color='#407cdb', label='Train Loss')
    ax.plot(history['val_loss'], color='#db5740', label='Validation Loss')
    ax.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('P茅rdida durante el entrenamiento')
    plt.show()


# Celda de c贸digo - Actividad 1
INPUT_FEATURES = 784 # 28*28
NUM_CLASSES = 11
N_EPOCHS = 50
LEARNING_RATE = 0.01

# 1. Define tu modelo
model_1 = Net(INPUT_FEATURES, NUM_CLASSES)
# (隆Puedes modificar la clase Net arriba para cambiar la arquitectura!)

# 2. Define el optimizador y la p茅rdida
optimizer_1 = optim.SGD(model_1.parameters(), lr=LEARNING_RATE)
criterion_1 = nn.CrossEntropyLoss()

# 3. Entrena
history_1 = train_model(model_1, train_loader, val_loader, optimizer_1, criterion_1, N_EPOCHS)

# 4. Grafica
plot_history(history_1)


# Celda de c贸digo - Actividad 2
def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 1. Define tu modelo
model_2 = Net(INPUT_FEATURES, NUM_CLASSES)
# 2. Aplica la inicializaci贸n
model_2.apply(weights_init_xavier)

# 3. Define el optimizador y la p茅rdida
optimizer_2 = optim.SGD(model_2.parameters(), lr=LEARNING_RATE)
criterion_2 = nn.CrossEntropyLoss()

# 4. Entrena
history_2 = train_model(model_2, train_loader, val_loader, optimizer_2, criterion_2, N_EPOCHS)

# 5. Grafica
plot_history(history_2)


#Celda de c贸digo - Actividad 3
# (Deber谩s modificar la clase Net para esta actividad
#  y luego instanciarla, optimizarla y entrenarla aqu铆)

model_3 = Net(INPUT_FEATURES, NUM_CLASSES)
optimizer_3 = optim.SGD(model_3.parameters(), lr=LEARNING_RATE)
criterion_3 = nn.CrossEntropyLoss()

history_3 = train_model(model_3, train_loader, val_loader, optimizer_3, criterion_3, N_EPOCHS)
plot_history(history_3)


class Net(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            # Capa 1: 784 -> 1024
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024), # Estabiliza el aprendizaje
            nn.ReLU(),
            
            # Capa 2: 1024 -> 512
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            # Capa 3: 512 -> 512
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # Capa 4: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # Capa 5: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Capa 6: Salida -> num_classes
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


# Celda de c贸digo - Actividad 4 (Prueba 1)
model_4_1 = Net(INPUT_FEATURES, NUM_CLASSES)
optimizer_4_1 = optim.SGD(model_4_1.parameters(), lr=LEARNING_RATE)
criterion_4_1 = nn.CrossEntropyLoss()
history_4_1 = train_model(model_4_1, train_loader, val_loader, optimizer_4_1, criterion_4_1, N_EPOCHS)
plot_history(history_4_1)


class Net(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


# Celda de c贸digo - Actividad 4 (Prueba 2)
model_4_2 = Net(INPUT_FEATURES, NUM_CLASSES)
optimizer_4_2 = optim.SGD(model_4_2.parameters(), lr=LEARNING_RATE)
criterion_4_2 = nn.CrossEntropyLoss()
history_4_2 = train_model(model_4_2, train_loader, val_loader, optimizer_4_2, criterion_4_2, N_EPOCHS)
plot_history(history_4_2)


# Celda de c贸digo - Actividad 5

# 1. Usa tu mejor arquitectura
model_5 = Net(INPUT_FEATURES, NUM_CLASSES) # (instancia de tu mejor Net)
# (Si usaste una inicializaci贸n custom, apl铆cala)
# model_5.apply(...)

# 2. Define el optimizador ADAM y la p茅rdida
# Prueba el learning rate por defecto de Adam (0.001)
optimizer_5 = optim.Adam(model_5.parameters(), lr=0.001)
criterion_5 = nn.CrossEntropyLoss()

# 3. Entrena
history_5 = train_model(model_5, train_loader, val_loader, optimizer_5, criterion_5, N_EPOCHS)

# 4. Grafica
plot_history(history_5)


# Celda de c贸digo - Actividad 6

# Elige tu mejor modelo (ej. model_5)
best_model = model_1

best_model.eval() # Modo evaluaci贸n
test_preds = []

with torch.no_grad():
    for images in test_loader:
        outputs = best_model(images)
        # Obtenemos el 铆ndice (la clase) con la probabilidad m谩s alta
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.numpy())

# --- Crear el archivo de env铆o ---
# El submission.csv debe tener "image_id" y "label"
# Asumimos que "image_id" es solo el 铆ndice (0, 1, 2...)
import pandas as pd

submission = pd.DataFrame({
    "Id": list(range(len(test_preds))),
    "label": test_preds
})

submission.to_csv("submission.csv", index=False)

print("隆Archivo submission.csv creado con 茅xito!")
print(submission.head())


len(submission)

