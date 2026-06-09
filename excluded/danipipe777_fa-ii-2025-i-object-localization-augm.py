#pip install -U albumentations


from __future__ import print_function, division

from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.typing import NDArray
from functools import reduce
from itertools import islice
import math
from itertools import chain
import copy
from PIL import Image

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
import torchvision 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchsummary import summary
# Import albumentations library in order to -use pre-built augmentations
import albumentations as A

from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count

import os
import torch
import os.path as osp
from skimage import io, transform
import matplotlib.pyplot as plt
import typing as ty
import cv2


plt.ion()   # interactive mode

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for root, dirs, filenames in os.walk('/kaggle/input'):
    for i, filepath in enumerate(filenames):
        if i >= 10:
            print()
            break
        print(osp.join(root, filepath))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


torch.manual_seed(32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')
test = torch.ones((100, 100)).to(device)
del test
torch.cuda.empty_cache()


DATA_DIR = '/kaggle/input/fa-ii-2025-i-object-localization/'
WORK_DIR = '/kaggle/working'
BATCH_SIZE = 32

img_dir = osp.join(DATA_DIR, "images/images")

df = pd.read_csv(osp.join(DATA_DIR, "train.csv"))

obj2id  = {'f16':0,'cougar':1,'chinook':2,'ah64':3,'f15':4,'seahawk':5}

id2obj  = {0:'f16',1:'cougar',2:'chinook',3:'ah64',4:'f15',5:'seahawk'}

df["class_id"] = df["class"].map(obj2id)

columns_f=['filename','xmin','ymin','xmax','ymax','class','class_id']

df= df[columns_f].copy()


df.head()


list_image = list(df.filename)
data_shape = []
data_dim = []
data_w = []
data_h = []

for i in tqdm(list_image): ## tqdm(list_image)dura 40 segundos
    ruta_imagen = osp.join(img_dir, i)
    imagen = io.imread(ruta_imagen)
    shapes = imagen.shape
    dimen = imagen.ndim
    imagen = Image.open(ruta_imagen)
    w, h = imagen.size
    data_w.append(w)
    data_h.append(h)
    data_shape.append(shapes)
    data_dim.append(dimen)

data_h_w = pd.DataFrame([list_image,data_shape,data_dim,data_w,data_h]).T.rename(columns={0:'filename',1:'shapes',2:'ndim',3:'w',4:'h'}) 
data_h_w = data_h_w.merge(df, how='inner', on='filename')


print("Las imagenes del dataset tienen la siguiente altura",data_h_w['h'].unique())
print("Las imagenes del dataset tienen la siguiente anchura",data_h_w['w'].unique())
print("Las imagenes del dataset tienen la siguiente dimension",data_h_w['ndim'].unique())



data_h_w['shapes'].value_counts()



plt.figure(figsize=(8,6))
sns.countplot(x=data_h_w['class'])
plt.title('Distribución de Clases')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()


data_h_w['width'] = data_h_w['xmax'] - data_h_w['xmin']
data_h_w['height'] = data_h_w['ymax'] - data_h_w['ymin']
data_h_w['area'] = data_h_w['width'] * data_h_w['height']


plt.figure(figsize=(10,6))
sns.histplot(data_h_w['area'], bins=30, kde=True)
plt.title('Distribución del Área de las Bounding Boxes')
plt.xlabel('Área (px^2)')
plt.ylabel('Frecuencia')
plt.show()


data_h_w['aspect_ratio'] = data_h_w['width'] / data_h_w['height']
sns.histplot(data_h_w['aspect_ratio'], bins=20, kde=True)
plt.title("Distribución del Ratio de Aspecto")
plt.show()


df[df['xmin']>=df['xmax']].shape, df[df['ymin']>=df['ymax']].shape
# No pueden presentarse coordenadas min con valores mayores a coordenadas max


out_of_bounds = data_h_w[(data_h_w['xmin'] < 0) | (data_h_w['ymin'] < 0) |
                   (data_h_w['xmax'] > data_h_w['w']) | 
                   (data_h_w['ymax'] > data_h_w['h'])]

print(f"Bounding Boxes fuera de rango: {len(out_of_bounds)}")


h_real=720
w_real=1280
h, w, c = 225, 400, 3 # The heigh, width and number of channels of each image


print(df[["ymin", "ymax", "xmin", "xmax"]].describe())


# Normalizar las columnas ymin, ymax, xmin, xmax
df[["ymin", "ymax"]] = df[["ymin", "ymax"]].div(h_real, axis=0)
df[["xmin", "xmax"]] = df[["xmin", "xmax"]].div(w_real, axis=0)



print(df[["ymin", "ymax", "xmin", "xmax"]].describe())


train_df, val_df = train_test_split(
    df, stratify=df['class_id'], test_size=0.35, random_state=42
)

print(train_df.shape)
print(val_df.shape)



transform_func_inp_signature = ty.Dict[str, NDArray[np.float_]]
transform_func_signature = ty.Callable[
    [transform_func_inp_signature],
    transform_func_inp_signature
]

class militarDataset(Dataset):
    """
    Location image dataset
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        root_dir: str, 
        labeled: bool = True,
        transform: ty.Optional[ty.List[transform_func_signature]] = None,
        output_size: ty.Optional[tuple] = None  # Añadir parámetro para tamaño de salida
    ) -> None:
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled
        self.output_size = output_size  # Almacenar el tamaño de salida
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> transform_func_signature: 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read image
        img_name = os.path.join(self.root_dir, self.df.filename.iloc[idx])
        #img_name = os.path.join(self.root_dir, self.df.iloc[idx]['filename'])
        image = io.imread(img_name)
        #image = cv2.imread(img_name)
        
        
        #print(f"Dimensiones originales de la imagen: {image.shape}")  # Agregar para depuración
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
            
        if image.ndim == 2:  # Si la imagen está en escala de grises
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)  # Convertir a RGB
        elif image.shape[2] == 4:  # Si la imagen es RGBA
            image = image[:, :, :3] 
            
        # Redimensionar la imagen si se especifica un tamaño de salida
        if self.output_size:
            image = cv2.resize(image, self.output_size)  # Redimensionar la imagen
        
        sample = {'image': image}
        
        if self.labeled:
            # Read labels
            img_class = self.df.class_id.iloc[idx]
            img_bbox = self.df.iloc[idx, 1:5]

            img_bbox = np.array([img_bbox]).astype('float')
            img_class = np.array([img_class]).astype('int')
            sample.update({'bbox': img_bbox, 'class_id': img_class})
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def draw_bbox(img, bbox, color,thickness: int = 3):
    xmin, ymin, xmax, ymax = bbox
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    return img

def desnormalize_bbox(bbox, h: int, w: int):
    """Escala las coordenadas normalizadas al tamaño real de la imagen."""
    return [
        int(bbox[0] * w),  # xmin
        int(bbox[1] * h),  # ymin
        int(bbox[2] * w),  # xmax
        int(bbox[3] * h),  # ymax
    ]

def draw_bboxes(imgs, bboxes, colors,thickness):
    """Dibuja múltiples cuadros delimitadores en imágenes, escalando según h y w."""
    for i, (img, bbox, color) in enumerate(zip(imgs, bboxes, colors)):
        imgs[i] = draw_bbox(img, bbox, color,thickness)
    return imgs

def draw_classes(imgs, classes, colors, origin, prefix: str ='',fontScale : int = 2):
    """Dibuja las clases en las imágenes."""
    for i, (img, class_id, color) in enumerate(zip(imgs, classes, colors)):
        if type(c)==list:
            name_class_=id2obj[classes[i]]
        else:
            name_class_=id2obj[classes[i][0]]
        imgs[i] = cv2.putText(
            img, f'{prefix}{name_class_}', #class_id.squeeze()
            origin, cv2.FONT_HERSHEY_SIMPLEX,
            fontScale , color, 2, cv2.LINE_AA
        )
    return imgs

def draw_predictions(imgs, classes, bboxes, colors, origin,thickness,fontScale):
    """
    Combina las funciones anteriores para dibujar cuadros delimitadores
    y clases en las imágenes.
    """
    assert all(len(x) > 0 for x in [imgs, classes, bboxes, colors])
    if len(colors) == 1:
        colors = [colors[0] for _ in imgs]
    imgs = draw_bboxes(imgs, bboxes, colors,thickness)
    imgs = draw_classes(imgs, classes, colors, origin,"",fontScale)
    return imgs


train_root_dir = osp.join(DATA_DIR, "images/images")#, "train"
train_ds = militarDataset(train_df, root_dir=train_root_dir,output_size=(w,h))

num_imgs = 6
start_idx = 0

samples = [train_ds[i] for i in range(start_idx, num_imgs)]

imgs = [s['image'] for s in samples]
bboxes = [desnormalize_bbox(s['bbox'].squeeze(),h,w) for s in samples]
classes = [s['class_id'] for s in samples]

imgs = draw_predictions(imgs, classes, bboxes, [(0, 150, 0)], (int(w*0.1), int(h*0.1)),thickness = 1,fontScale=1)#(150, 10)

fig = plt.figure(figsize=(30, num_imgs))

for i, img in enumerate(imgs):
    fig.add_subplot(1, num_imgs, i+1)
    plt.imshow(img)

plt.show()


train_ds = militarDataset(train_df, root_dir=train_root_dir,output_size=(w,h))#,output_size=(255,255)

means = np.zeros(3)
stds = np.zeros(3)
n_images = 0

for x in train_ds:
    img = x['image']#.astype(np.float32)  # Asegúrate de que la imagen está en float para cálculos precisos
    n_images += 1

    for channel in range(3):
        channel_pixels = img[..., channel]
        # Acumular la suma y suma de cuadrados para calcular la media y desviación estándar
        means[channel] += np.mean(channel_pixels)
        stds[channel] += np.std(channel_pixels)

# Calcular la media y desviación estándar final
means /= n_images
stds /= n_images


print(means)
print(stds)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C (0,1,2)
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        sample.update({'image': image})
        return sample


class Normalizer(object):
    
    def __init__(self, stds, means):
        """
        Arguments:
        
            stds: array of length 3 containing the standard deviation of each channel in RGB order.
            means: array of length 3 containing the means of each channel in RGB order.
        """
        self.stds = stds
        self.means = means
    
    def __call__(self, sample):
        """
        Sample: a dicitonary containing:
            image: sample image in format (C, H, W)
        Returns:
            the image in (C, H, W) format with the channels normalized.
        """
        image = sample['image']
        
        for channel in range(3):
            image[channel] = (image[channel] - means[channel]) / stds[channel]

        sample['image'] = image
        return sample

class TVTransformWrapper(object):
    """Torch Vision Transform Wrapper
    """
    def __init__(self, transform: torch.nn.Module):
        self.transform = transform
        
    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample

class AlbumentationsWrapper(object):
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, sample):
        transformed = self.transform(
            image=sample['image'], 
            bboxes=sample['bbox']
        )
        sample['image'] = transformed['image']
        sample['bbox'] = np.array(transformed['bboxes'])
        return sample



train_data_augmentations = A.Compose([
    A.HorizontalFlip(p=1)    
    ],
    bbox_params=A.BboxParams(
        format='albumentations',
        label_fields=[],#'category_ids'
    )
)

dataaug_transforms = torchvision.transforms.Compose(
    [
        AlbumentationsWrapper(train_data_augmentations)
    ]
)



import shutil

if os.path.exists('data_final'):
    shutil.rmtree('data_final')

os.mkdir('data_final')

train_ds_da = militarDataset(train_df, root_dir=train_root_dir) # ,output_size=(255,255) No usamos resize para que conservé el tamaño original
last_index = max(train_ds_da.df.filename.apply(lambda x: int(x.replace('image_', '').replace('.jpeg', ''))))
index = last_index + 1
rows = []
for j in range(0,1): #Cantidad de imagenes que se quieran generar por cada imagen original del train
    iterador = iter(train_ds_da)
    for i in range(len(train_ds_da)): 
        x = next(iterador) 
        x_transformed = copy.deepcopy(x) 
        x_transformed = dataaug_transforms(x_transformed)
        filename = f"image_id_{index}_t{j}.jpeg"
        image = x_transformed['image']#.astype('uint8')
        #result = cv2.normalize(image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite("data_final/"+filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #row = [filename, [id2obj[class_id] for class_id in x_transformed["class_id"]], *x_transformed['bbox'].squeeze()]
        row = [filename, *x_transformed["class_id"], *x_transformed['bbox'].squeeze()] 
        rows.append(row) 
        index+=1

aug_df = pd.DataFrame(rows, columns=['filename', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax',])
#aug_df["class"] = aug_df.class_id.replace(train_df.set_index('class_id')['class'])
#aug_df[['xmin', 'ymin', 'xmax', 'ymax']]*=hw_real


#A esta carpeta creada con las imágenes sintéticas se agregan las originales para tener todo en una sola carpeta
source = train_root_dir
destination = 'data_final'

#gather all files
allfiles = os.listdir(source)

#iterate on all files to move them to destination folder
for f in allfiles:
    if f in train_df['filename'].values:
        src_path = os.path.join(source, f) 
        dst_path = os.path.join(destination, f) 
        shutil.copy(src_path, dst_path)

#
dataframe_with_dataaugmentation = pd.concat([train_df, aug_df], ignore_index=True)
dataframe_with_dataaugmentation['class']=dataframe_with_dataaugmentation['class_id'].replace(id2obj)
dataframe_with_dataaugmentation


allfiles = os.listdir('data_final')

#iterate on all files to move them to destination folder
len(allfiles)


common_transforms = [
    ToTensor(),
    Normalizer(
        means=means,
        stds=stds,
    )
]

train_data_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5)
    
    ],
    bbox_params=A.BboxParams(
        format='albumentations', 
        label_fields=[],
    )
)

train_transforms = torchvision.transforms.Compose(
    [
        AlbumentationsWrapper(train_data_augmentations),
    ] + common_transforms
)

eval_transforms = torchvision.transforms.Compose(common_transforms)




train_ds = militarDataset(dataframe_with_dataaugmentation, root_dir='data_final', transform=train_transforms,output_size=(w,h)) #train_root_dir
train_data = torch.utils.data.DataLoader(train_ds, batch_size=16)

for x in train_data:
    print(x['image'].size())
    break


train_ds = militarDataset(train_df, root_dir=train_root_dir,output_size=(w,h))#,output_size=(255,255)

x = next(iter(train_ds))
x_transformed = copy.deepcopy(x)
x_transformed = train_transforms(x_transformed)

original_img = x['image']
transformed_img = x_transformed['image'].numpy().transpose(1, 2, 0)
#transformed_img = cv2.resize(transformed_img, (255,255)) ## Redimensiona la imagen

original_img = draw_bbox(
    original_img,
    desnormalize_bbox(x['bbox'].squeeze(),h,w),
    (0, 255, 0),thickness=1
)

transformed_img = draw_bbox(
    transformed_img,
    desnormalize_bbox(x_transformed['bbox'].squeeze(),h,w),
    (0, 255, 0),thickness=1
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].imshow(original_img)
axes[0].set_title('Original digit')

axes[1].imshow(transformed_img)
axes[1].set_title('Transformed digit')

plt.show()


from torchvision.models import vgg16


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
  
    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.dropout(out) 
        return out 

# Load the vgg16 model
vgg16_model = vgg16(pretrained=True, progress=True)
pretrained_model = FeatureExtractor(vgg16_model).to(device)
#pretrained_model


vgg16_model


def get_output_shape(model: nn.Sequential, image_dim: ty.Tuple[int, int, int]):
    return model(torch.rand(*(image_dim)).to(device)).data.shape

class Model(nn.Module):
    def __init__(self, input_shape: ty.Tuple[int, int, int] = (3, 255, 400), n_classes: int = 6):
        """
        Model with one input (image) and two outputs: 
            1. Digit classification (classification).
            2. Bounding box prediction (regression). 
        
        Arguments:
            input_shape: input shape of the image in format (C, H, W)
            n_classes: number of classes to perfrom classification with
            
        Attributes:
            backbone: ConvNet that process the image and 
            returns a flattened vector with the information of the 
            activations.
            
            cls_head: MLP that receives the flattened input from the backbone 
            and predicts the classification logits for the classes (classficiation task).
            
            reg_head: MLP that receives the flattened input from the backbone 
            and predicts the coordinates of the predicted bounding box (regression task). 
        """
        super().__init__()
        
        self.input_shape = input_shape
        
        # When doing transfer learning, use pretrained model instead of custom backbone
        self.backbone = pretrained_model
        
        backbone_output_shape = get_output_shape(self.backbone, [1, *input_shape])
        backbone_output_features = reduce(lambda x, y: x*y, backbone_output_shape)
        
        self.cls_head = nn.Sequential(
            nn.Linear(in_features=backbone_output_features, out_features=256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(in_features=backbone_output_features, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x: Tensor) -> ty.Dict[str, Tensor]:
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        pred_bbox = self.reg_head(features)
        predictions = {'bbox': pred_bbox, 'class_id': cls_logits}
        return predictions


train_ds = militarDataset(dataframe_with_dataaugmentation, root_dir='data_final', transform=train_transforms,output_size=(w,h)) #train_root_dir
train_data = torch.utils.data.DataLoader(train_ds, batch_size=16)

for x in train_data:
    print(x['image'].size())
    break


print('image', x['image'].size())
model = Model(input_shape=(3, h, w), n_classes=6).to(device)
x['image'] = x['image'].to(device)
preds = model(x['image'])
preds


def iou(y_true: Tensor, y_pred: Tensor):
    pairwise_iou = torchvision.ops.box_iou(y_true.squeeze(), y_pred.squeeze())
    result = torch.trace(pairwise_iou) / pairwise_iou.size()[0]
    return result


def accuracy(y_true: Tensor, y_pred: Tensor):
    pred = torch.argmax(y_pred, axis=-1)
    y_true = y_true.squeeze()
    correct = torch.eq(pred, y_true).float()
    total = torch.ones_like(correct)
    result = torch.divide(torch.sum(correct), torch.sum(total))
    return result


def loss_fn(y_true, y_preds, alpha: float = 0.5):
    cls_y_true, cls_y_pred = y_true['class_id'].long(), y_preds['class_id'].float().unsqueeze(-1)
    reg_y_true, reg_y_pred = y_true['bbox'].float().squeeze(), y_preds['bbox'].float().squeeze()
    
    cls_loss = F.cross_entropy(cls_y_pred, cls_y_true)
    
    reg_loss = F.mse_loss(reg_y_pred, reg_y_true)
    # Adds weights to both tasks
    total_loss = (1 - alpha) * cls_loss + alpha * reg_loss
    return dict(loss=total_loss, reg_loss=reg_loss,cls_loss=cls_loss)


def printer(logs: ty.Dict[str, ty.Any]):
    # print every 10 steps
    if logs['iters'] % 10 != 0:
        return
    print('Iteration #: ',logs['iters'])
    for name, value in logs.items():
        if name == 'iters':
            continue
        
        if type(value) in [float, int]:
            value = round(value, 4)
        elif type(value) is torch.Tensor:
            value = torch.round(value, decimals=4)
        
        print(f'\t{name} = {value}')
    print()


def evaluate(
    logs: ty.Dict[str, ty.Any], 
    labels: ty.Dict[str, Tensor],
    preds: ty.Dict[str, Tensor],
    eval_set: str,
    metrics: ty.Dict[str, ty.Callable[[Tensor, Tensor], Tensor]],
    losses: ty.Optional[ty.Dict[str, Tensor]] = None,
) -> ty.Dict[str, ty.Any]:
    
    if losses is not None:
        for loss_name, loss_value in losses.items():
            logs[f'{eval_set}_{loss_name}'] = loss_value
    
    for task_name, label in labels.items():
        for metric_name, metric in metrics[task_name]:
            value = metric(label, preds[task_name])
            logs[f'{eval_set}_{metric_name}'] = value
            
    return logs

def step(
    model: Model, 
    optimizer: Optimizer, 
    batch: militarDataset,
    loss_fn: ty.Callable[[ty.Dict[str, torch.Tensor]], torch.Tensor],
    device: str,
    train: bool = False,
) -> ty.Tuple[ty.Dict[str, Tensor], ty.Dict[str, Tensor]]:
    
    if train:
        optimizer.zero_grad()
    
    #img = batch['image'].to(device)
    img = batch.pop('image').to(device)
    
    for k in list(batch.keys()):
        batch[k] = batch[k].to(device)
    
    preds = model(img.float())
    losses = loss_fn(batch, preds)
    final_loss = losses['loss']
    
    if train:
        final_loss.backward()
        optimizer.step()
    
    return losses, preds


def train(
    model: Model, 
    optimizer: Optimizer, 
    dataset: DataLoader,
    eval_datasets: ty.List[ty.Tuple[str, DataLoader]],
    loss_fn: ty.Callable[[ty.Dict[str, torch.Tensor]], torch.Tensor],
    metrics: ty.Dict[str, ty.Callable[[Tensor, Tensor], Tensor]],
    callbacks: ty.List[ty.Callable[[ty.Dict[ty.Any, ty.Any]], None]],
    device: str,
    train_steps: 100,
    eval_steps: 10,
) -> Model:
    # Send model to device (GPU or CPU)
    model = model.to(device)
    iters = 0
    iterator = iter(dataset)
    assert train_steps > eval_steps, 'Train steps should be greater than the eval steps'
    
    while iters <= train_steps:
        logs = dict()
        logs['iters'] = iters
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            batch = next(iterator)
        # Send batch to device 
        losses, preds = step(model, optimizer, batch, loss_fn, device, train=True)
        logs = evaluate(logs, batch, preds, 'train', metrics, losses)
        
        # Eval every eval_steps iterations
        if iters % eval_steps == 0:        
            # Evaluate
            # Deactives layers that only needed to train
            # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
            model.eval()
            
            # Avoids calculating gradients in evaluation dataset. 
            with torch.no_grad():

                for name, dataset in eval_datasets:
                    
                    for batch in dataset:
                        losses, preds = step(model, optimizer, batch, loss_fn, device, train=False)            
                        logs = evaluate(logs, batch, preds, name, metrics, losses)
        
        for callback in callbacks:
            callback(logs)
        
        iters += 1
    
    return model




torch.cuda.empty_cache()


# Hparams
batch_size = 16
lr = 0.001

# Data
train_ds = militarDataset(dataframe_with_dataaugmentation, root_dir='data_final', transform=train_transforms,output_size=(w,h))#,output_size=(255,255)
val_ds = militarDataset(val_df, root_dir=train_root_dir, transform=eval_transforms,output_size=(w,h)) #,output_size=(255,255)

train_data = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
val_data = DataLoader(val_ds, batch_size=batch_size, num_workers=cpu_count())

# Model
model = Model().to(device)
summary(model, model.input_shape)


# Optimizer
optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

model = train(
    model,
    optimizer,
    train_data,
    eval_datasets=[('val', val_data)],
    loss_fn=loss_fn,
    metrics={
        'bbox': [('iou', iou)],
        'class_id': [('accuracy', accuracy)]
    },
    callbacks=[printer],
    device=device,
    train_steps=100,
    eval_steps=10
)


# Save the model to disk
torch.save(model, 'pretrained_model.pth')


num_imgs = 5
ncols = 5
nrows = math.ceil(num_imgs / ncols)

start_idx = 0

inference_ds = militarDataset(val_df.iloc[start_idx:start_idx+num_imgs], root_dir=train_root_dir,output_size=(w,h))#,output_size=(255,255)
inference_data = DataLoader(inference_ds, batch_size=num_imgs, num_workers=1, shuffle=False)
inference_batch = next(iter(inference_data))
inference_imgs = np.empty((num_imgs, 3, h, w))

transform = eval_transforms

for i, img in enumerate(inference_batch['image']):
    inference_imgs[i] = transform(dict(image=img.numpy()))['image'].numpy()

preds = model(torch.tensor(inference_imgs).float().to(device))

samples = [inference_ds[i] for i in range(start_idx, num_imgs)]

imgs = [s['image'] for s in samples]
bboxes = [desnormalize_bbox(s['bbox'].squeeze(), h, w) for s in samples]
classes = [s['class_id'] for s in samples]

pred_bboxes = preds['bbox'].detach().cpu().numpy()
pred_bboxes = [desnormalize_bbox(bbox, h, w) for bbox in pred_bboxes]
pred_classes = preds['class_id'].argmax(-1).detach().cpu().numpy()


# Green: ground truth

imgs = draw_predictions(imgs, classes, bboxes, [(0, 150, 0)], (int(w*0.1), int(h*0.1)),thickness = 1,fontScale=1)
# Red: predicted
pred_classes_ = []
for i in range(num_imgs):
    temp = np.array([pred_classes[i]])
    pred_classes_.append(temp)
imgs = draw_predictions(imgs, pred_classes_, pred_bboxes, [(200, 0, 0)], (int(w*0.8), int(h*0.8)),thickness = 1,fontScale=1)

fig, axes = plt.subplots(nrows=nrows, ncols=num_imgs // nrows, figsize=(30, 30))

for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i])
    ax.axis('off')

plt.tight_layout()
plt.show()



# Perform inference on cpu in order to avoid memory problems 
device = 'cuda'
model = model.to(device)

test_root_dir = osp.join(DATA_DIR, "images/images")
test_df = pd.read_csv(osp.join(DATA_DIR, "test.csv"))

test_ds = militarDataset(test_df, root_dir=test_root_dir, labeled=False, transform=eval_transforms,output_size=(w,h))#
test_data = DataLoader(test_ds, batch_size=1, num_workers=cpu_count(), shuffle=False)

class_preds = []
bbox_preds = []

for batch in test_data:
    batch_preds = model(batch['image'].float().to(device))
    
    class_pred = batch_preds['class_id'].argmax(-1).detach().cpu().numpy()
    bbox_pred = batch_preds['bbox'].detach().cpu().numpy()
    
    class_preds.append(class_pred.squeeze())
    bbox_preds.append(bbox_pred.squeeze())


class_preds = np.array(class_preds)
bbox_preds = np.array(bbox_preds)


submission = pd.DataFrame(
    index=test_df.filename,
    data={
        'class_id': class_preds,
        }
)
submission


submission["xmin"] = bbox_preds[:, 0]*w_real
submission["ymin"] = bbox_preds[:, 1]*h_real
submission["xmax"] = bbox_preds[:, 2]*w_real
submission["ymax"] = bbox_preds[:, 3]*h_real


submission['class']=submission['class_id'].replace(id2obj)


submission.rename(columns={'class_id':'class'}, inplace=True)


submission.to_csv('submission.csv')

