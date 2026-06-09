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


classes_encoded


PATCH_SIZE = 16
CLIP_IMG_SIZE = 224
NUM_PATCHES = CLIP_IMG_SIZE // PATCH_SIZE


import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_clip_representation(images, processor, model, device='cpu'):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(outputs, dim=-1)


from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def generate_seg_masks(imgs_path, classes_encoded, processor, model, device='cuda'):
    imgs_names = []
    heatmaps = []

    for img_name in tqdm(os.listdir(imgs_path)):
        img = Image.open(os.path.join(imgs_path, img_name))
        img_shapes = np.array(img).shape

        # Resize image
        img_resized = img.resize((224, 224))
        image_encoded = get_clip_representation([img_resized], processor, model, device)

        # Compute class similarities
        class_similarities = image_encoded @ classes_encoded.T
        chosen_class_num = class_similarities.argmax(axis=1)[0]
        chosen_class_emb = classes_encoded[chosen_class_num].to(device)

        img = np.array(img_resized)
        PATCH_SIZE = 32
        NUM_PATCHES = 224 // PATCH_SIZE

        # Extract patches
        img_patches = [
            Image.fromarray(img[x:x+PATCH_SIZE, y:y+PATCH_SIZE]) 
            for x in range(0, 224, PATCH_SIZE) 
            for y in range(0, 224, PATCH_SIZE)
        ]

        img_patches_embs = get_clip_representation(img_patches, processor, model, device)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        heatmap = img_patches_embs_sims.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0), scale_factor=PATCH_SIZE, mode='bilinear'
        ).squeeze().to(device)

        # Normalize and threshold
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        threshold = filters.threshold_otsu(heatmap.detach().cpu().numpy())
        increased_threshold = threshold * 1.0897
        heatmap = (heatmap > increased_threshold).float()

        # Apply Gaussian filtering
        heatmap = gaussian_filter(heatmap.cpu().numpy(), sigma=1)

        # Morphological operations
        heatmap = morphology.opening(heatmap, morphology.disk(3))
        heatmap = morphology.closing(heatmap, morphology.disk(3))

        # Active contour model
        s = np.linspace(0, 2*np.pi, 400)
        x = 112 + 100*np.cos(s)
        y = 112 + 100*np.sin(s)
        init_snake = np.array([x, y]).T
        snake = active_contour(heatmap, init_snake, alpha=0.1, beta=10, gamma=0.01)

        # Scale the heatmap back to original image size
        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), interpolation=transforms.InterpolationMode.NEAREST)
        ])
        final_mask = target_transform(torch.tensor(heatmap).unsqueeze(0)).squeeze().numpy()

        imgs_names.append(img_name)
        heatmaps.append(final_mask)

    return imgs_names, heatmaps




val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded, processor, model, device='cuda')


import matplotlib.pyplot as plt

NUM_VIS_IMG = 10
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
test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, classes_encoded, processor, model, device='cuda')


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

