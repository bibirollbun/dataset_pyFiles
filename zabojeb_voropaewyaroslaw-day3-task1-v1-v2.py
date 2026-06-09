class_prompts = {
    'american_bulldog': "a strong, muscular American Bulldog",
    'basset_hound': "a Basset Hound with long ears",
    'keeshond': "a fluffy Keeshond dog",
    'British_Shorthair': "a chubby British Shorthair cat",
    'Sphynx': "a hairless Sphynx cat",
    'pomeranian': "a small, fluffy Pomeranian dog",
    'Egyptian_Mau': "a spotted Egyptian Mau cat",
    'Birman': "a blue-eyed Birman cat",
    'american_pit_bull_terrier': "an athletic American Pit Bull Terrier",
    'japanese_chin': "a small Japanese Chin dog",
    'Maine_Coon': "a large, bushy-tailed Maine Coon cat",
    'beagle': "a Beagle with a curious expression",
    'Bombay': "a sleek, black Bombay cat",
    'wheaten_terrier': "a Wheaten Terrier with a soft coat",
    'shiba_inu': "a confident Shiba Inu dog",
    'havanese': "a small, fluffy Havanese dog",
    'miniature_pinscher': "a Miniature Pinscher with a sleek body",
    'yorkshire_terrier': "a tiny Yorkshire Terrier with long hair",
    'boxer': "a playful Boxer dog",
    'scottish_terrier': "a Scottish Terrier with a distinctive profile",
    'newfoundland': "a large, gentle Newfoundland dog",
    'chihuahua': "a tiny Chihuahua with big eyes",
    'saint_bernard': "a giant Saint Bernard dog",
    'Persian': "a Persian cat with long fur",
    'Bengal': "a Bengal cat with a spotted coat",
    'german_shorthaired': "a German Shorthaired Pointer",
    'english_cocker_spaniel': "an English Cocker Spaniel with long ears",
    'leonberger': "a massive Leonberger dog",
    'Siamese': "a Siamese cat with blue eyes",
    'Abyssinian': "a sleek Abyssinian cat",
    'staffordshire_bull_terrier': "a stocky Staffordshire Bull Terrier",
    'Ragdoll': "a Ragdoll cat with a relaxed demeanor",
    'pug': "a Pug with a wrinkled face",
    'Russian_Blue': "a Russian Blue cat with a dense coat",
    'samoyed': "a Samoyed dog with a fluffy white coat",
    'english_setter': "an English Setter with a speckled coat",
    'great_pyrenees': "a Great Pyrenees dog with a thick coat"
}

class_names = list(class_prompts.values())


from transformers import AutoProcessor, CLIPModel, CLIPFeatureExtractor

device = 'cuda:0'
model_name = "openai/clip-vit-base-patch16"

processor = AutoProcessor.from_pretrained(model_name)

model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()


import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# получаем эмбеддинги названий классов
classes = [' '.join(x.lower().split('_')) for x in class_names]
classes_tokenized = processor(text=class_names, images=None, return_tensors="pt", padding=True).to(device)
classes_encoded = model.get_text_features(**classes_tokenized)
classes_encoded = F.normalize(classes_encoded, dim=-1)

PATCH_SIZE = 32
CLIP_IMG_SIZE = 224
NUM_PATCHES = CLIP_IMG_SIZE // PATCH_SIZE

def get_clip_representation(imgs):
    with torch.no_grad():
        inputs = processor(text='', images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        images_encoded = F.normalize(outputs.image_embeds, dim=-1)

    return images_encoded

def generate_seg_masks(imgs_path, classes_encoded, mhv):
    imgs_names = []
    heatmaps = []

    for img_name in tqdm(os.listdir(imgs_path)):
        # загружаем картинку
        img = Image.open(os.path.join(imgs_path, img_name))

        # сохраняем оригинальные размеры картинки
        img_shapes = np.array(img).shape
        
        # получаем clip эмбеддинг целой картинки
        image_encoded = get_clip_representation(img)

        # определяем, что за порода на этой картинке
        chosen_class_num = (image_encoded @ classes_encoded.T).argmax(axis=1)[0]

        # получаем эмбеддинг названия породы
        chosen_class_emb = classes_encoded[chosen_class_num]

        # ресайзим картинку и разбиваем на патчи 
        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE,y:y+PATCH_SIZE]) for x in range(0,CLIP_IMG_SIZE,PATCH_SIZE) for y in range(0,224,PATCH_SIZE)] # hardco

        # вычисляем косинусное сходство между эмбеддингами патчами and эмбеддингом класса
        img_patches_embs = get_clip_representation(img_patches)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        # формируем heatmap
        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)

        # интерполируем heatmap до размера (CLIP_IMG_SIZE, CLIP_IMG_SIZE)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                            scale_factor=PATCH_SIZE,
                                            mode='bilinear').to(device)

        # порог между FG(объектом) and BG(фоном) будет просто средним значением heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # !!!
        # mean_heatmap_value = heatmap.mean() # !!!
        mean_heatmap_value = mhv

        # получаем карту сегментации
        heatmap = heatmap.ge(mean_heatmap_value).type(heatmap.type())

        # ресайзим карту сегментации обратно до размеров исходной картинки
        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), Image.NEAREST),
        ])
        heatmap = target_transform(heatmap).data.cpu().numpy()

        # сохраняем полученную карту сегментации для текущей картинки 
        imgs_names.append(img_name)
        heatmaps.append(heatmap[0][0])

    return imgs_names, heatmaps


val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'


def binaryMaskIOU(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1==1,  mask2==1))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


best_metric = -float("inf")
best_value = 0.5
for i in range(0, 1000, 25):
    m = i / 1000
    val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded, m)
    
    val_ious = []
    for img_name, seg_mask in zip(val_img_names, val_seg_masks):
    
        mask = Image.open(os.path.join(val_masks_path, img_name.replace('.jpg', '.png')))
        mask = np.array(mask)//255
        iou = binaryMaskIOU(seg_mask, mask)
        val_ious.append(iou)

    metric = np.mean(val_ious)

    if metric > best_metric:
        print(f"NEW BEST: {m} with score {metric}")
        best_metric = metric
        best_value = m


val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded, 0.5)


import matplotlib.pyplot as plt

for NUM_VIS_IMG in range(5):
    f, axes = plt.subplots(1,3)
    axes[0].imshow(np.array(Image.open(os.path.join(val_imgs_path, 
                                                    val_img_names[NUM_VIS_IMG]
                                                   )
                                      )
                           )
                  )
    axes[1].imshow(val_seg_masks[NUM_VIS_IMG])
    axes[2].imshow(np.array(Image.open(os.path.join(val_masks_path, 
                                                    val_img_names[NUM_VIS_IMG].replace('jpg', 'png')
                                                   )
                                      )
                            )
                  )


def binaryMaskIOU(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1==1,  mask2==1))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


val_ious = []
for img_name, seg_mask in zip(val_img_names, val_seg_masks):

    mask = Image.open(os.path.join(val_masks_path, img_name.replace('.jpg', '.png')))
    mask = np.array(mask)//255
    iou = binaryMaskIOU(seg_mask, mask)
    val_ious.append(iou)


np.mean(val_ious) # previous best: 0.4611868874551629


test_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/test_imgs'
test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, classes_encoded, 0.5)


from io import BytesIO
import base64
import pandas as pd
import hashlib

def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """ Конвертирует картинку PIL.Image в base64 (текстовый формат). """
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

ids = []
b64 = []

for img_name, seg_mask in zip(test_img_names, test_seg_masks):
    ids.append(img_name[:-4]) # get rid og .jpg part
    mask = Image.fromarray(255*seg_mask)
    b64.append(image_to_base64(mask.convert("L")))

df = pd.DataFrame({"img_id": [int(id_) for id_ in ids], "mask": b64})
hsh = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
submit_path = f"submit_{hsh}.csv"
print(f"SUBMIT_NAME: {submit_path}")
print(df.head(10))
df.to_csv(submit_path,index=False)


Я решаю задание:
Вам даны изображения кошек и собак разных пород. Ваша задача — сегментировать существ на них, т. е. создать бинарную карту сегментации для каждого изображения. Данные делятся на:
- validation set, содержащий 20 изображений кошек/собак и соответствующие им бинарные карты сегментации
- test set, содержащий 1000 изображений кошек и собак. Вам необходимо создать карты сегментации для них.
Правила:
- Разрешено создавать любые промпты для CLIP.
- Вы можете делать все, что угодно, включая обучение на валидационных и тестовых данных (удачи с этим)))

За исключением этого, вам дается список пород кошек и собак, которые присутствуют в данных. Однако не гарантируется, что каждая порода присутствует в валидационных данных. Вот они:

class_names = ['american_bulldog',
 'basset_hound',
 'keeshond',
 'British_Shorthair',
 'Sphynx',
 'pomeranian',
 'Egyptian_Mau',
 'Birman',
 'american_pit_bull_terrier',
 'japanese_chin',
 'Maine_Coon',
 'beagle',
 'Bombay',
 'wheaten_terrier',
 'shiba_inu',
 'havanese',
 'miniature_pinscher',
 'yorkshire_terrier',
 'boxer',
 'scottish_terrier',
 'newfoundland',
 'chihuahua',
 'saint_bernard',
 'Persian',
 'Bengal',
 'german_shorthaired',
 'english_cocker_spaniel',
 'leonberger',
 'Siamese',
 'Abyssinian',
 'staffordshire_bull_terrier',
 'Ragdoll',
 'pug',
 'Russian_Blue',
 'samoyed',
 'english_setter',
 'great_pyrenees']

Основной код базового решения:

import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# получаем эмбеддинги названий классов
classes = [' '.join(x.lower().split('_')) for x in class_names]
classes_tokenized = processor(text=class_names, images=None, return_tensors="pt", padding=True).to(device)
classes_encoded = model.get_text_features(**classes_tokenized)
classes_encoded = F.normalize(classes_encoded, dim=-1)

PATCH_SIZE = 16
CLIP_IMG_SIZE = 224
NUM_PATCHES = CLIP_IMG_SIZE // PATCH_SIZE

def get_clip_representation(imgs):
    
    with torch.no_grad():
        inputs = processor(text='', images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        images_encoded = F.normalize(outputs.image_embeds, dim=-1)

    return images_encoded

def generate_seg_masks(imgs_path, classes_encoded):

    imgs_names = []
    heatmaps = []

    for img_name in tqdm(os.listdir(imgs_path)):

        # загружаем картинку
        img = Image.open(os.path.join(imgs_path, img_name))

        # сохраняем оригинальные размеры картинки
        img_shapes = np.array(img).shape
        

        # получаем clip эмбеддинг целой картинки
        image_encoded = get_clip_representation(img)

        # определяем, что за порода на этой картинке
        chosen_class_num = (image_encoded @ classes_encoded.T).argmax(axis=1)[0]
        # получаем эмбеддинг названия породы
        chosen_class_emb = classes_encoded[chosen_class_num]

        # ресайзим картинку и разбиваем на патчи 
        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE,y:y+PATCH_SIZE]) for x in range(0,CLIP_IMG_SIZE,PATCH_SIZE) for y in range(0,224,16)]

        # вычисляем косинусное сходство между эмбеддингами патчами and эмбеддингом класса
        img_patches_embs = get_clip_representation(img_patches)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        # формируем heatmap
        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
        # интерполируем heatmap до размера (CLIP_IMG_SIZE, CLIP_IMG_SIZE)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                            scale_factor=PATCH_SIZE,
                                            mode='bilinear').to(device)

        # порог между FG(объектом) and BG(фоном) будет просто средним значением heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        mean_heatmap_value = heatmap.mean()
        # получаем карту сегментации
        heatmap = heatmap.ge(mean_heatmap_value).type(heatmap.type())

        # ресайзим карту сегментации обратно до размеров исходной картинки
        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), Image.NEAREST),
        ])
        heatmap = target_transform(heatmap).data.cpu().numpy()

        # сохраняем полученную карту сегментации для текущей картинки 
        imgs_names.append(img_name)
        heatmaps.append(heatmap[0][0])

    return imgs_names, heatmaps

val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded)

Это решение работает плохо. Помоги мне и напиши улучшенное решение, которое даст хороший score




Task:
Вам даны изображения кошек и собак разных пород. Ваша задача — сегментировать существ на них, т. е. создать бинарную карту сегментации для каждого изображения. Данные делятся на:

    validation set, содержащий 20 изображений кошек/собак и соответствующие им бинарные карты сегментации
    test set, содержащий 1000 изображений кошек и собак. Вам необходимо создать карты сегментации для них.
    Вам разрешено использовать только одну предобученную модель CLIP для решения этой задачи. Вот полные ПРАВИЛА:
    Нельзя использовать никакие предварительно обученные модели, кроме данного CLIP
    Нельзя использовать никакие внешние наборы данных
    Разрешено создавать любые промпты для CLIP!!!
   РАЗРЕШЕНО ДООБУЧАТЬ МОДЕЛЬ НА ВАЛИДАЦИОННЫХ ДАННЫХ!

Prompts:
class_names = ['american_bulldog',
'basset_hound',
'keeshond',
'British_Shorthair',
'Sphynx',
'pomeranian',
'Egyptian_Mau',
'Birman',
'american_pit_bull_terrier',
'japanese_chin',
'Maine_Coon',
'beagle',
'Bombay',
'wheaten_terrier',
'shiba_inu',
'havanese',
'miniature_pinscher',
'yorkshire_terrier',
'boxer',
'scottish_terrier',
'newfoundland',
'chihuahua',
'saint_bernard',
'Persian',
'Bengal',
'german_shorthaired',
'english_cocker_spaniel',
'leonberger',
'Siamese',
'Abyssinian',
'staffordshire_bull_terrier',
'Ragdoll',
'pug',
'Russian_Blue',
'samoyed',
'english_setter',
'great_pyrenees']

Code:
from transformers import AutoProcessor, CLIPModel, CLIPFeatureExtractor

device = 'cuda:0'
model_name = "openai/clip-vit-base-patch16"

processor = AutoProcessor.from_pretrained(model_name)

model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# получаем эмбеддинги названий классов
classes = [' '.join(x.lower().split('_')) for x in class_names]
classes_tokenized = processor(text=class_names, images=None, return_tensors="pt", padding=True).to(device)
classes_encoded = model.get_text_features(**classes_tokenized)
classes_encoded = F.normalize(classes_encoded, dim=-1)

PATCH_SIZE = 16
CLIP_IMG_SIZE = 224
NUM_PATCHES = CLIP_IMG_SIZE // PATCH_SIZE

def get_clip_representation(imgs):
    with torch.no_grad():
        inputs = processor(text='', images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        images_encoded = F.normalize(outputs.image_embeds, dim=-1)

    return images_encoded

def generate_seg_masks(imgs_path, classes_encoded):
    imgs_names = []
    heatmaps = []

    for img_name in tqdm(os.listdir(imgs_path)):
        # загружаем картинку
        img = Image.open(os.path.join(imgs_path, img_name))

        # сохраняем оригинальные размеры картинки
        img_shapes = np.array(img).shape
        
        # получаем clip эмбеддинг целой картинки
        image_encoded = get_clip_representation(img)

        # определяем, что за порода на этой картинке
        chosen_class_num = (image_encoded @ classes_encoded.T).argmax(axis=1)[0]

        # получаем эмбеддинг названия породы
        chosen_class_emb = classes_encoded[chosen_class_num]

        # ресайзим картинку и разбиваем на патчи 
        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE,y:y+PATCH_SIZE]) for x in range(0,CLIP_IMG_SIZE,PATCH_SIZE) for y in range(0,224,PATCH_SIZE)] # hardco

        # вычисляем косинусное сходство между эмбеддингами патчами and эмбеддингом класса
        img_patches_embs = get_clip_representation(img_patches)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        # формируем heatmap
        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)

        # интерполируем heatmap до размера (CLIP_IMG_SIZE, CLIP_IMG_SIZE)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                            scale_factor=PATCH_SIZE,
                                            mode='bilinear').to(device)

        # порог между FG(объектом) and BG(фоном) будет просто средним значением heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # !!!
        # mean_heatmap_value = heatmap.mean() # !!!
        mean_heatmap_value = heatmap.mean()

        # получаем карту сегментации
        heatmap = heatmap.ge(mean_heatmap_value).type(heatmap.type())

        # ресайзим карту сегментации обратно до размеров исходной картинки
        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), Image.NEAREST),
        ])
        heatmap = target_transform(heatmap).data.cpu().numpy()

        # сохраняем полученную карту сегментации для текущей картинки 
        imgs_names.append(img_name)
        heatmaps.append(heatmap[0][0])

    return imgs_names, heatmaps

val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded)

Write a full new code with fine-tuning CLIP on validation set

