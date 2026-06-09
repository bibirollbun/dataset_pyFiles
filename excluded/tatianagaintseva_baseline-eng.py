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

        # loading image
        img = Image.open(os.path.join(imgs_path, img_name))

        # save original img width and height
        img_shapes = np.array(img).shape
        

        # get clip embedding of the whole image
        image_encoded = get_clip_representation(img)

        # determine class of the image
        chosen_class_num = (image_encoded @ classes_encoded.T).argmax(axis=1)[0]
        # get embedding of the chosen class description
        chosen_class_emb = classes_encoded[chosen_class_num]

        # resize image and split it into tiles 
        img = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
        img_patches = [Image.fromarray(img[x:x+PATCH_SIZE,y:y+PATCH_SIZE]) for x in range(0,CLIP_IMG_SIZE,PATCH_SIZE) for y in range(0,224,16)]

        # get cosine similarities between patches embeddings and class embedding
        img_patches_embs = get_clip_representation(img_patches)
        img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T

        # form a heatmap
        heatmap = img_patches_embs_sims
        heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
        # interpolate heatmap into size (CLIP_IMG_SIZE, CLIP_IMG_SIZE)
        heatmap = torch.nn.functional.interpolate(heatmap[:, np.newaxis],
                                            scale_factor=PATCH_SIZE,
                                            mode='bilinear').to(device)

        # threshold between FG(foreground) and BG(background) is the mean value of a heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        mean_heatmap_value = heatmap.mean()
        # get segmentation map 
        heatmap = heatmap.ge(mean_heatmap_value).type(heatmap.type())

        # resize segmentation map back to original size
        target_transform = transforms.Compose([
            transforms.Resize((img_shapes[0], img_shapes[1]), Image.NEAREST),
        ])
        heatmap = target_transform(heatmap).data.cpu().numpy()

        # save generated heatmap of a current image
        imgs_names.append(img_name)
        heatmaps.append(heatmap[0][0])

    return imgs_names, heatmaps


val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs/'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks/'

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

pred_pd = pd.DataFrame({"img_id": [int(id_) for id_ in ids], "mask": b64})
pred_pd.to_csv('test_submission.csv', index=False)

