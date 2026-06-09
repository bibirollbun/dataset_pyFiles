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


from transformers import AutoProcessor, CLIPModel, CLIPFeatureExtractor

device = 'cuda:0'
model_name = "openai/clip-vit-base-patch16"

processor = AutoProcessor.from_pretrained(model_name)

model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()


import torch
from torch.nn import functional as F


# Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ¸ Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğ¹ ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ²
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


from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def generate_seg_masks(imgs_path, classes_encoded):

    imgs_names = []
    heatmaps = []

    for img_name in tqdm(os.listdir(imgs_path)):

        # Ğ·Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºÑƒ
        img = Image.open(os.path.join(imgs_path, img_name))

        # Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¾Ñ€Ğ¸Ğ³Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ñ‹ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºĞ¸
        img_shapes = np.array(img).shape
        

        # Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ clip Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³ Ñ†ĞµĞ»Ğ¾Ğ¹ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºĞ¸
        image_encoded = get_clip_representation(img)

        # Ğ¾Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ�ĞµĞ¼, Ñ‡Ñ‚Ğ¾ Ğ·Ğ° Ğ¿Ğ¾Ñ€Ğ¾Ğ´Ğ° Ğ½Ğ° Ñ�Ñ‚Ğ¾Ğ¹ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºĞµ
        chosen_class_num = (image_encoded @ classes_encoded.T).argmax(axis=1)[0]
        # Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³ Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ñ� Ğ¿Ğ¾Ñ€Ğ¾Ğ´Ñ‹
        chosen_class_emb = classes_encoded[chosen_class_num]

        # Ñ€ĞµÑ�Ğ°Ğ¹Ğ·Ğ¸Ğ¼ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºÑƒ Ğ¸ Ñ€Ğ°Ğ·Ğ±Ğ¸Ğ²Ğ°ĞµĞ¼ Ğ½Ğ° Ğ¿Ğ°Ñ‚Ñ‡Ğ¸ 
        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE,y:y+PATCH_SIZE]) for x in range(0,CLIP_IMG_SIZE,PATCH_SIZE) for y in range(0,224,16)]

        # Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ ĞºĞ¾Ñ�Ğ¸Ğ½ÑƒÑ�Ğ½Ğ¾Ğµ Ñ�Ñ…Ğ¾Ğ´Ñ�Ñ‚Ğ²Ğ¾ Ğ¼ĞµĞ¶Ğ´Ñƒ Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ°Ğ¼Ğ¸ Ğ¿Ğ°Ñ‚Ñ‡Ğ°Ğ¼Ğ¸ and Ñ�Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ¾Ğ¼ ĞºĞ»Ğ°Ñ�Ñ�Ğ°
        img_patches_embs = get_clip_representation(img_patches)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        # Ñ„Ğ¾Ñ€Ğ¼Ğ¸Ñ€ÑƒĞµĞ¼ heatmap
        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
        # Ğ¸Ğ½Ñ‚ĞµÑ€Ğ¿Ğ¾Ğ»Ğ¸Ñ€ÑƒĞµĞ¼ heatmap Ğ´Ğ¾ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ° (CLIP_IMG_SIZE, CLIP_IMG_SIZE)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                            scale_factor=PATCH_SIZE,
                                            mode='bilinear').to(device)

        # Ğ¿Ğ¾Ñ€Ğ¾Ğ³ Ğ¼ĞµĞ¶Ğ´Ñƒ FG(Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ¼) and BG(Ñ„Ğ¾Ğ½Ğ¾Ğ¼) Ğ±ÑƒĞ´ĞµÑ‚ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾ Ñ�Ñ€ĞµĞ´Ğ½Ğ¸Ğ¼ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸ĞµĞ¼ heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        mean_heatmap_value = heatmap.mean()
        # Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ ĞºĞ°Ñ€Ñ‚Ñƒ Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚Ğ°Ñ†Ğ¸Ğ¸
        heatmap = heatmap.ge(mean_heatmap_value).type(heatmap.type())

        # Ñ€ĞµÑ�Ğ°Ğ¹Ğ·Ğ¸Ğ¼ ĞºĞ°Ñ€Ñ‚Ñƒ Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚Ğ°Ñ†Ğ¸Ğ¸ Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ğ¾ Ğ´Ğ¾ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ¾Ğ² Ğ¸Ñ�Ñ…Ğ¾Ğ´Ğ½Ğ¾Ğ¹ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºĞ¸
        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), Image.NEAREST),
        ])
        heatmap = target_transform(heatmap).data.cpu().numpy()

        # Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ½ÑƒÑ� ĞºĞ°Ñ€Ñ‚Ñƒ Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚Ğ°Ñ†Ğ¸Ğ¸ Ğ´Ğ»Ñ� Ñ‚ĞµĞºÑƒÑ‰ĞµĞ¹ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºĞ¸ 
        imgs_names.append(img_name)
        heatmaps.append(heatmap[0][0])

    return imgs_names, heatmaps


val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded)


import matplotlib.pyplot as plt

NUM_VIS_IMG = 0
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


np.mean(val_ious)


test_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/test_imgs'
test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, classes_encoded)


from io import BytesIO
import base64
import pandas as pd
import hashlib

def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """ ĞšĞ¾Ğ½Ğ²ĞµÑ€Ñ‚Ğ¸Ñ€ÑƒĞµÑ‚ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºÑƒ PIL.Image Ğ² base64 (Ñ‚ĞµĞºÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğ¹ Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚). """
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

