# Crear un DataFrame de pandas
df_submission = pd.DataFrame({
    'Id': ids,
    'Category': predictions
})

# Guardar como archivo CSV sin el Ã­ndice de pandas
df_submission.to_csv('mi_submission.csv', index=False)

print("Â¡Archivo 'mi_submission.csv' creado con Ã©xito!")


# Celda de cÃ³digo
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


# Celda de cÃ³digo
# (Ajuste clave)
# Esta clase de Dataset estÃ¡ adaptada para PyTorch.
# Nota: nn.CrossEntropyLoss espera Ã­ndices de clase (0, 1, 2...)
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
        # Las imÃ¡genes son 28x28, las convertimos a 1x28x28 (canal, alto, ancho)
        image = self._x[idx].reshape(1, 28, 28)

        # Convertimos a torch.Tensor de tipo float
        image = torch.tensor(image, dtype=torch.float32)

        if self._transform is not None:
            image = self._transform(image)

        if self._y is None:
            return image

        # Devolvemos el Ã­ndice de la clase como un LongTensor
        label = torch.tensor(self._y[idx], dtype=torch.long)
        return image, label


# Celda de cÃ³digo
# FunciÃ³n de normalizaciÃ³n (Â¡importante para las redes neuronales!)
def normalize(tensor_image):
    # Normaliza entre 0 y 1 (ya que los datos originales van de 0 a 255)
    return tensor_image / 255.0


import os
os.listdir("/kaggle/input/hackathon-de-ia-unsch")


data_train = np.load('/kaggle/input/hackathon-de-ia-unsch/train_set.npz', allow_pickle=True)
data_val = np.load('/kaggle/input/hackathon-de-ia-unsch/val_set.npz',allow_pickle=True)
data_test = np.load('/kaggle/input/hackathon-de-ia-unsch/test_set.npz',allow_pickle=True)


# Extraer imÃ¡genes y etiquetas
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



# Celda de cÃ³digo
class Net(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Net, self).__init__()
        
        # Bloque Convolucional 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 1 canal (gris) -> 32 filtros
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 28x28 -> 14x14
        )
        
        # Bloque Convolucional 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 14x14 -> 7x7
        )

        # Capas de ClasificaciÃ³n (MLP al final)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # RegularizaciÃ³n fuerte para evitar overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Si la imagen llega aplanada (784), la volvemos a poner en (1, 28, 28)
        if len(x.shape) == 2:
            x = x.view(-1, 1, 28, 28)
            
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        logits = self.classifier(x)
        return logits


# Celda de cÃ³digo
def train_epoch(model, loader, optimizer, criterion):
    model.train() # Pone el modelo en modo entrenamiento
    running_loss = 0.0

    for images, labels in loader:
        # 1. Poner gradientes a cero
        optimizer.zero_grad()

        # 2. Forward pass (obtener predicciones)
        outputs = model(images)

        # 3. Calcular la pÃ©rdida
        loss = criterion(outputs, labels)

        # 4. Backward pass (calcular gradientes)
        loss.backward()

        # 5. Actualizar pesos
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate_epoch(model, loader, criterion):
    model.eval() # Pone el modelo en modo evaluaciÃ³n
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Desactiva el cÃ¡lculo de gradientes
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Guardar predicciones para la mÃ©trica
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # Usamos la mÃ©trica de la competiciÃ³n: Balanced Accuracy
    acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, acc


# Celda de cÃ³digo
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

# FunciÃ³n para graficar la pÃ©rdida
def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], color='#407cdb', label='Train Loss')
    ax.plot(history['val_loss'], color='#db5740', label='Validation Loss')
    ax.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PÃ©rdida durante el entrenamiento')
    plt.show()


import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_features, num_classes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        
        # Arquitectura Profunda (Actividad 4.1) con ReLU (Actividad 4.3)
        self.layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512), # Evita que el aprendizaje se pierda
            nn.ReLU(),
            nn.Dropout(0.3),     # Evita el sobreajuste
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# FunciÃ³n de InicializaciÃ³n (Actividad 4.2)
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(784, 11).to(device)
model.apply(weights_init)


# Celda de cÃ³digo - Actividad 2
def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 1. Define tu modelo
model_2 = Net(INPUT_FEATURES, NUM_CLASSES)
# 2. Aplica la inicializaciÃ³n
model_2.apply(weights_init_xavier)

# 3. Define el optimizador y la pÃ©rdida
optimizer_2 = optim.SGD(model_2.parameters(), lr=LEARNING_RATE)
criterion_2 = nn.CrossEntropyLoss()

# 4. Entrena
history_2 = train_model(model_2, train_loader, val_loader, optimizer_2, criterion_2, N_EPOCHS)

# 5. Grafica
plot_history(history_2)


#Celda de cÃ³digo - Actividad 3
# (DeberÃ¡s modificar la clase Net para esta actividad
#  y luego instanciarla, optimizarla y entrenarla aquÃ­)

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
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


# Celda de cÃ³digo - Actividad 4 (Prueba 1)
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


# Celda de cÃ³digo - Actividad 4 (Prueba 2)
model_4_2 = Net(INPUT_FEATURES, NUM_CLASSES)
optimizer_4_2 = optim.SGD(model_4_2.parameters(), lr=LEARNING_RATE)
criterion_4_2 = nn.CrossEntropyLoss()
history_4_2 = train_model(model_4_2, train_loader, val_loader, optimizer_4_2, criterion_4_2, N_EPOCHS)
plot_history(history_4_2)




# --- Celda de Actividad 5 ---

# 1. Definir hiperparÃ¡metros bÃ¡sicos
INPUT_FEATURES = 784 # (28x28)
NUM_CLASSES = 11
N_EPOCHS = 30  # Subimos a 30 para darle tiempo a aprender

# 2. Instanciar tu mejor modelo (la clase Net que definiste antes)
# AsegÃºrate de haber corrido antes la celda donde definiste la clase Net (CNN o MLP profunda)
model_5 = Net(INPUT_FEATURES, NUM_CLASSES).to(device)

# 3. Definir el optimizador ADAM y la pÃ©rdida
# Adam es mucho mÃ¡s eficiente que SGD para llegar a 0.800
optimizer_5 = optim.Adam(model_5.parameters(), lr=0.001)
criterion_5 = nn.CrossEntropyLoss()

# 4. Entrenar el modelo
# Esta funciÃ³n ya estÃ¡ definida en tus celdas anteriores
history_5 = train_model(model_5, train_loader, val_loader, optimizer_5, criterion_5, N_EPOCHS)

# 5. Graficar los resultados para ver si hubo overfitting
plot_history(history_5)


# --- Celda de Actividad 6: Generar el archivo final ---

# 1. Usamos el modelo que entrenamos en la Actividad 5
model_5.eval() 
test_preds = []

# 2. Realizar las predicciones sobre el test_loader
with torch.no_grad():
    for images in test_loader:
        # Movemos las imÃ¡genes al mismo dispositivo que el modelo (GPU o CPU)
        images = images.to(device)
        
        # Obtenemos la salida del modelo
        outputs = model_5(images)
        
        # Elegimos la clase con mayor probabilidad
        _, preds = torch.max(outputs, 1)
        
        # Guardamos los resultados (moviÃ©ndolos a la CPU para pandas)
        test_preds.extend(preds.cpu().numpy())

# 3. Crear el DataFrame con el formato que pide la competencia
import pandas as pd

submission = pd.DataFrame({
    "Id": list(range(len(test_preds))), # Genera IDs de 0 a N
    "label": test_preds
})

# 4. Guardar el archivo CSV
# Nota: AsegÃºrate de que el nombre sea "submission.csv" si la plataforma lo pide asÃ­
submission.to_csv("submission.csv", index=False)

print("âœ… Â¡Archivo 'submission.csv' creado con Ã©xito!")
print(f"Total de predicciones: {len(submission)}")
print(submission.head()) # Muestra las primeras 5 filas para verificar


len(submission)

