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


from transformers import AutoProcessor, CLIPModel
import torch
import torch.nn as nn

device = 'cuda:0'
model_name = "openai/clip-vit-base-patch16"

processor = AutoProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

class AdditionalLayers(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdditionalLayers, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Use model.vision_model.embeddings to get input dimension
input_dim = model.vision_model.config.hidden_size

# Create additional layers
additional_layers = AdditionalLayers(input_dim=input_dim, output_dim=10)

class ExtendedCLIP(nn.Module):
    def __init__(self, clip_model, additional_layers):
        super(ExtendedCLIP, self).__init__()
        self.clip = clip_model
        self.additional_layers = additional_layers

    def forward(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        additional_output = self.additional_layers(pooled_output)
        return additional_output

# Create extended model
extended_model = ExtendedCLIP(clip_model=model, additional_layers=additional_layers).to(device)


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


import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os

def post_process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def apply_dynamic_threshold(heatmap):
    threshold = heatmap.mean() + heatmap.std()
    dynamic_mask = (heatmap > threshold).astype(np.uint8)
    return dynamic_mask

def apply_thresholding_methods(heatmap):
    heatmap = (heatmap * 255).astype(np.uint8)
    
    _, otsu_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_gaussian_mask = cv2.adaptiveThreshold(heatmap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)
    adaptive_mean_mask = cv2.adaptiveThreshold(heatmap, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
    _, triangle_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    simple_mask = (heatmap > 127).astype(np.uint8)
    dynamic_mask = apply_dynamic_threshold(heatmap)
    
    # Canny edge detection
    canny_edges = cv2.Canny(heatmap, 100, 200)
    
    return {
        'otsu': otsu_mask / 255,
        'adaptive_gaussian': adaptive_gaussian_mask / 255,
        'adaptive_mean': adaptive_mean_mask / 255,
        'triangle': triangle_mask / 255,
        'simple': simple_mask,
        'dynamic': dynamic_mask,
        'canny': canny_edges / 255
    }

def generate_seg_masks(imgs_path, classes_encoded):
    imgs_names = []
    heatmaps = []

    for img_name in tqdm(os.listdir(imgs_path)):
        img = Image.open(os.path.join(imgs_path, img_name))
        img_shapes = np.array(img).shape

        image_encoded = get_clip_representation(img)
        chosen_class_num = (image_encoded @ classes_encoded.T).argmax(axis=1)[0]
        chosen_class_emb = classes_encoded[chosen_class_num]

        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE, y:y+PATCH_SIZE]) 
                       for x in range(0, CLIP_IMG_SIZE, PATCH_SIZE) 
                       for y in range(0, 224, 16)]

        img_patches_embs = get_clip_representation(img_patches)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                                  scale_factor=PATCH_SIZE,
                                                  mode='bilinear').to(device)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        # Apply thresholding methods
        masks = apply_thresholding_methods(heatmap)

        # Choose a mask to post-process
        final_mask = post_process_mask(masks['otsu'])  # You can switch between masks

        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), transforms.InterpolationMode.NEAREST),
        ])
        final_mask = target_transform(Image.fromarray((final_mask * 255).astype(np.uint8))).convert('L')
        final_mask = np.array(final_mask) / 255

        imgs_names.append(img_name)
        heatmaps.append(final_mask)

    return imgs_names, heatmaps

def binaryMaskIOU(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou

# Example usage
val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'  # Replace with actual path
val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded)



val_ious = {method: [] for method in apply_thresholding_methods(np.zeros((1, 1))).keys()}



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


import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os

# Define constants
CLIP_IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = CLIP_IMG_SIZE // PATCH_SIZE

def post_process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def apply_dynamic_threshold(heatmap):
    threshold = heatmap.mean() + heatmap.std()
    dynamic_mask = (heatmap > threshold).astype(np.uint8)
    return dynamic_mask

def apply_thresholding_methods(heatmap):
    heatmap = (heatmap * 255).astype(np.uint8)
    _, otsu_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_gaussian_mask = cv2.adaptiveThreshold(heatmap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)
    adaptive_mean_mask = cv2.adaptiveThreshold(heatmap, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
    _, triangle_mask = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    simple_mask = (heatmap > 127).astype(np.uint8)
    dynamic_mask = apply_dynamic_threshold(heatmap)
    canny_edges = cv2.Canny(heatmap, 100, 200)

    return {
        'otsu': otsu_mask / 255,
        'adaptive_gaussian': adaptive_gaussian_mask / 255,
        'adaptive_mean': adaptive_mean_mask / 255,
        'triangle': triangle_mask / 255,
        'simple': simple_mask,
        'dynamic': dynamic_mask,
        'canny': canny_edges / 255
    }


from sklearn.preprocessing import LabelEncoder
import numpy as np

def generate_seg_masks(imgs_path, class_names):
    label_encoder = LabelEncoder()
    classes_encoded = label_encoder.fit_transform(class_names)
    
    imgs_names = []
    heatmaps = []
    

    for img_name in tqdm(os.listdir(imgs_path)):
        img = Image.open(os.path.join(imgs_path, img_name))
        img_shapes = np.array(img).shape

        # Use the extended model to get embeddings
        inputs = processor(images=[img], return_tensors="pt").to(device)
        image_encoded = extended_model(pixel_values=inputs['pixel_values']).cpu().detach().numpy()
        
        chosen_class_num = (image_encoded @ classes_encoded).argmax(axis=1)[0]
        chosen_class_emb = classes_encoded[chosen_class_num]

        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE, y:y+PATCH_SIZE])
                       for x in range(0, CLIP_IMG_SIZE, PATCH_SIZE)
                       for y in range(0, 224, 16)]

        img_patches_embs = extended_model(pixel_values=processor(images=img_patches, return_tensors="pt")['pixel_values'].to(device)).cpu().detach().numpy()
        
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                                  scale_factor=PATCH_SIZE,
                                                  mode='bilinear').to(device)

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.squeeze().cpu().detach().numpy()

        # Apply thresholding methods
        masks = apply_thresholding_methods(heatmap)

        # Choose a mask to post-process
        final_mask = post_process_mask(masks['otsu'])

        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), transforms.InterpolationMode.NEAREST),
        ])
        final_mask = target_transform(Image.fromarray((final_mask * 255).astype(np.uint8))).convert('L')
        final_mask = np.array(final_mask) / 255

        imgs_names.append(img_name)
        heatmaps.append(final_mask)
    return imgs_names, heatmaps



def binaryMaskIOU(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou

# Example usage
val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks' 

def encode_classes(class_names):
    return {name: idx for idx, name in enumerate(class_names)}

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
class_encodings = encode_classes(class_names)

input_values = input()
image_encoded = extended_model(pixel_values=input_values['pixel_values']).cpu().detach().numpy()
image_encoded = extended_model(pixel_values=input['pixel_values']).cpu().detach().numpy()
print("image_encoded shape:", image_encoded.shape)

classes_encoded = np.array(classes_encoded)
print("classes_encoded shape:", classes_encoded.shape)
# Pass class_encodings.values() or a similar structure to your function
val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, list(class_encodings.values()))

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, class_names)

val_ious = {method: [] for method in apply_thresholding_methods(np.zeros((1, 1))).keys()}

for img_name, seg_mask in zip(val_img_names, val_seg_masks):
    mask = Image.open(os.path.join(val_masks_path, img_name.replace('.jpg', '.png')))
    mask = np.array(mask) // 255

    for method, generated_mask in masks.items():
        iou = binaryMaskIOU(generated_mask, mask)
        val_ious[method].append(iou)

# Calculate mean IoU for each method
mean_ious = {method: np.mean(iou_list) for method, iou_list in val_ious.items()}
print(mean_ious)

test_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/test_imgs'
test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, class_names)

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
    ids.append(img_name[:-4])  # get rid of .jpg part
    mask = Image.fromarray(255*seg_mask)
    b64.append(image_to_base64(mask.convert("L")))

df = pd.DataFrame({"img_id": [int(id_) for id_ in ids], "mask": b64})
hsh = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
submit_path = f"submit_{hsh}.csv"
print(f"SUBMIT_NAME: {submit_path}")
print(df.head(10))
df.to_csv(submit_path, index=False)

