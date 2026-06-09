#Importamos las librerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import gc
import os
import time
import random
from datetime import datetime
from PIL import Image
from tqdm.notebook import tqdm
from sklearn import model_selection, metrics
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import seaborn as sns


# Tomamos los datos de Kaggle y los cargamos en un DataFrame

Data = "/kaggle/input/cassava-leaf-disease-classification"
Train = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
Test = "/kaggle/input/cassava-leaf-disease-classification/test_images/"


df = pd.read_csv(os.path.join(Data, "train.csv"))


os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"

# Para la reproducibilidad de los resultados hacemos uso de una semilla fija
# Esto asegura que los resultados sean consistentes en diferentes ejecuciones

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


IMG_SIZE = 224

BATCH_SIZE = 16

LR = 2e-05

N_EPOCHS = 10


# Veamos como son los datos 
df = pd.read_csv(os.path.join(Data, "train.csv"))

# Realizamos la separacion de train y test:
train_df, valid_df = model_selection.train_test_split(df, 
                                                      test_size=0.1, 
                                                      random_state=42, 
                                                      stratify=df.label.values)

df.head()




# Graficamos el diagrama de barras
df.label.value_counts().plot(kind="bar")

label_counts = df.label.value_counts()

# Graficamos el pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2", len(label_counts)))

plt.title('Distribución de Etiquetas en el Conjunto de Datos', fontsize=16)

plt.show()



class CassavaDataset(torch.utils.data.Dataset):

    def __init__(self, df, data_path= Data, mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        self.data_dir = "train_images" if mode == "train" else "test_images"

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(img)

        return image, label


# Tiempo para realizar aumentos de datos (Data Augmentation)

# Conjunto de transformaciones para el conjunto de entrenamiento
transforms_train = transforms.Compose(
[
    # Redimensiona la imagen al tamaño previamente especificado en IMG_SIZE
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    
    # Voltea aleatoriamente las imágenes de manera horizontal con una probabilidad del 30%
    transforms.RandomHorizontalFlip(p=0.3),
    
    # Voltea aleatoriamente las imágenes de manera vertical con una probabilidad del 30%
    transforms.RandomVerticalFlip(p=0.3),
    
    # Rota aleatoriamente las imágenes hasta 10 grados
    transforms.RandomRotation(10),
    
    # Aplica una transformación afín aleatoria de hasta 10 grados
    transforms.RandomAffine(10),
    
    # Recorta aleatoriamente las imágenes y las redimensiona al tamaño especificado por IMG_SIZE
    transforms.RandomResizedCrop(IMG_SIZE),
    
    # Convierte la imagen en un tensor, adecuado para el modelo en PyTorch
    transforms.ToTensor(),
    
    # Normaliza la imagen usando la media y desviación estándar de los canales RGB
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]
)

# Conjunto de transformaciones para el conjunto de validación
transforms_valid = transforms.Compose(
[
    # Redimensiona la imagen al tamaño especificado en IMG_SIZE
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    
    # Convierte la imagen en un tensor
    transforms.ToTensor(),
    
    # Normaliza la imagen utilizando la media y desviación estándar de los canales RGB
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]
)



class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False, model_path=None):
        super(ViTBase16, self).__init__()

        # Crear el modelo Vision Transformer
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        # Cargar pesos preentrenados si se solicita
        if pretrained:
            if model_path is not None:
                self.model.load_state_dict(torch.load(model_path))
            else:
                raise ValueError("Se requiere un modelo preentrenado y una ruta de archivo si se activa `pretrained`.")

        # Ajustar la última capa para que tenga el número correcto de clases
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        # Pasar las imágenes por el modelo para obtener las predicciones
        return self.model(x)

    def train_one_epoch(self, train_loader, criterion, optimizer, device, lr_scheduler=None):
        # Inicializar variables de pérdida y precisión
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        # Establecer el modelo en modo de entrenamiento
        self.model.train()

        for i, (data, target) in enumerate(train_loader):
            # Enviar los datos al dispositivo adecuado (CPU o GPU)
            data, target = data.to(device), target.to(device)

            # Limpiar los gradientes del optimizador
            optimizer.zero_grad()

            # Paso hacia adelante: calcular las predicciones del modelo
            output = self.forward(data)

            # Calcular la pérdida
            loss = criterion(output, target)

            # Paso hacia atrás: calcular los gradientes de la pérdida con respecto a los parámetros del modelo
            loss.backward()

            # Calcular la precisión
            accuracy = (output.argmax(dim=1) == target).float().mean()

            # Acumular la pérdida y precisión de la época
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

            # Realizar un paso de optimización
            optimizer.step()

            # Actualizar la tasa de aprendizaje si se usa un scheduler
            if lr_scheduler:
                lr_scheduler.step()

            # Imprimir información sobre el progreso
            if i % 20 == 0:
                print(f"\tBATCH {i+1}/{len(train_loader)} - LOSS: {loss.item():.4f} - ACC: {accuracy.item():.4f}")

        # Promediar la pérdida y precisión para la época
        return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)

    def validate_one_epoch(self, valid_loader, criterion, device):
        # Inicializar variables para la pérdida y precisión de validación
        valid_loss = 0.0
        valid_accuracy = 0.0

        # Establecer el modelo en modo de evaluación (sin gradientes)
        self.model.eval()

        with torch.no_grad():  # Desactivar el cálculo de gradientes para la validación
            for data, target in valid_loader:
                # Enviar los datos al dispositivo adecuado
                data, target = data.to(device), target.to(device)
    
                # Paso hacia adelante: calcular las predicciones
                output = self.model(data)
    
                # Calcular la pérdida
                loss = criterion(output, target)
    
                # Calcular la precisión
                accuracy = (output.argmax(dim=1) == target).float().mean()
    
                # Acumular la pérdida y precisión de validación
                valid_loss += loss.item()
                valid_accuracy += accuracy.item()

        # Promediar la pérdida y precisión para la validación
        return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)


def fit_tpu(model, epochs, device, criterion, optimizer, train_loader, valid_loader=None):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    
    valid_loss_min = np.inf
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []

    for epoch in range(1, epochs + 1):
        gc.collect()
        para_train_loader = pl.ParallelLoader(train_loader, [device])

        xm.master_print(f"{'='*50}")
        xm.master_print(f"EPOCH {epoch} - TRAINING...")

        train_loss, train_acc = model.train_one_epoch(
            para_train_loader.per_device_loader(device), criterion, optimizer, device
        )
        
        xm.master_print(f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss:.4f}, ACCURACY: {train_acc:.4f}\n")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        if valid_loader is not None:
            gc.collect()
            para_valid_loader = pl.ParallelLoader(valid_loader, [device])
            xm.master_print(f"EPOCH {epoch} - VALIDATING...")

            with torch.no_grad():
                valid_loss, valid_acc = model.validate_one_epoch(
                    para_valid_loader.per_device_loader(device), criterion, device
                )
                
            xm.master_print(f"\t[VALID] LOSS: {valid_loss:.4f}, ACCURACY: {valid_acc:.4f}\n")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)

            if valid_loss < valid_loss_min:
                xm.master_print(f"Validation loss improved ({valid_loss_min:.4f} --> {valid_loss:.4f}). Saving model...")
                xm.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
                valid_loss_min = valid_loss

        xm.mark_step()
        gc.collect()

    return {
        "train_loss": train_losses,
        "valid_loss": valid_losses,
        "train_acc": train_accs,
        "valid_acc": valid_accs,
    }


def _run():
    import torch_xla.core.xla_model as xm
    
    try:
        # Inicializar el dispositivo TPU con una mejor gestión de errores
        device = xm.xla_device()
        xm.master_print(f"Successfully initialized TPU device: {device}")
        xm.master_print(f"World size: {xm.xrt_world_size()}")
        xm.master_print(f"Local rank: {xm.get_ordinal()}")
        
    except Exception as e:
        print(f"Failed to initialize TPU: {e}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
    
    # Crear el modelo
    model = ViTBase16(n_classes=5, pretrained=False)
    model.to(device)
    
    # crear los datasets
    train_dataset = CassavaDataset(train_df, transforms=transforms_train)
    valid_dataset = CassavaDataset(valid_df, transforms=transforms_valid)

    # Comprobar si estamos utilizando TPU formación distribuida
    if str(device).startswith('xla'):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )

        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
        )
        
        lr_multiplier = xm.xrt_world_size()
    else:
        train_sampler = None
        valid_sampler = None
        lr_multiplier = 1

    # DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=2,  
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=2,
    )

    # Optimizador y criterios
    criterion = nn.CrossEntropyLoss()
    lr = LR * lr_multiplier
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Informacion de entrenamiento
    print(f"Device: {device}")
    print(f"Learning Rate: {lr}")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    # Entrenar
    logs = fit_tpu(
        model=model,
        epochs=N_EPOCHS,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    # Guardar modelo final
    end_time = datetime.now()
    print(f"Training completed in: {end_time - start_time}")
    
    final_model_path = f'model_final_{end_time.strftime("%Y%m%d_%H%M")}.pth'
    if str(device).startswith('xla'):
        xm.save(model.state_dict(), final_model_path)
        xm.master_print(f"Final model saved as: {final_model_path}")
    else:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved as: {final_model_path}")


def _mp_fn(rank, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    _run()


if __name__ == "__main__":
    print("Attempting TPU training...")
    
    # TPU multiprocessing
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
        FLAGS = {}
        print("Starting TPU multiprocessing training...")
        xmp.spawn(_mp_fn, args=(FLAGS,), start_method="fork")
    except Exception as e:
        print(f"TPU multiprocessing failed: {e}")
        
        # Opcion 2: single TPU core
        try:
            print("Trying single TPU core...")
            _run()
        except Exception as e2:
            print(f"Single TPU failed: {e2}")
            
            # Opcion 3: CPU/GPU
            print("Falling back to CPU/GPU training...")
            _run()

