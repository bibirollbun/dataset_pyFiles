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


# test_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/test_imgs'
# test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, classes_encoded)


# from io import BytesIO
# import base64
# import pandas as pd
# import hashlib

# def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
#     """ ĞšĞ¾Ğ½Ğ²ĞµÑ€Ñ‚Ğ¸Ñ€ÑƒĞµÑ‚ ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºÑƒ PIL.Image Ğ² base64 (Ñ‚ĞµĞºÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğ¹ Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚). """
#     buf = BytesIO()
#     image.save(buf, format=fmt)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")

# ids = []
# b64 = []

# for img_name, seg_mask in zip(test_img_names, test_seg_masks):
#     ids.append(img_name[:-4]) # get rid og .jpg part
#     mask = Image.fromarray(255*seg_mask)
#     b64.append(image_to_base64(mask.convert("L")))

# df = pd.DataFrame({"img_id": [int(id_) for id_ in ids], "mask": b64})
# hsh = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
# submit_path = f"submit_{hsh}.csv"
# print(f"SUBMIT_NAME: {submit_path}")
# print(df.head(10))
# df.to_csv(submit_path,index=False)


import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import base64
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import binary_dilation
import torchvision.transforms as transforms
import hashlib
device = 'cuda'
model_name = "openai/clip-vit-base-patch16"
processor = AutoProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()



class_names = [
    'american_bulldog', 'basset_hound', 'keeshond', 'British_Shorthair', 'Sphynx',
    'pomeranian', 'Egyptian_Mau', 'Birman', 'american_pit_bull_terrier', 'japanese_chin',
    'Maine_Coon', 'beagle', 'Bombay', 'wheaten_terrier', 'shiba_inu', 'havanese',
    'miniature_pinscher', 'yorkshire_terrier', 'boxer', 'scottish_terrier', 'newfoundland',
    'chihuahua', 'saint_bernard', 'Persian', 'Bengal', 'german_shorthaired',
    'english_cocker_spaniel', 'leonberger', 'Siamese', 'Abyssinian',
    'staffordshire_bull_terrier', 'Ragdoll', 'pug', 'Russian_Blue', 'samoyed',
    'english_setter', 'great_pyrenees'
]

def get_text_embeddings():
    classes = [f"a photo of a {' '.join(x.lower().split('_'))}" for x in class_names]
    additional_prompts = ["a photo of a cat or dog", "a photo of background"]
    all_prompts = classes + additional_prompts
    with torch.no_grad():
        tokenized = processor(text=all_prompts, images=None, return_tensors="pt", padding=True).to(device)
        text_embeds = model.get_text_features(**tokenized)
        text_embeds = F.normalize(text_embeds, dim=-1)
    return text_embeds, len(classes)


def get_clip_activations(image):
    with torch.no_grad():
        inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model.vision_model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[-1][:, 1:, :]  # [batch, 196, 768]
        activations = F.normalize(activations, dim=-1)
    return activations

def binaryMaskIOU(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
    union = mask1_area + mask2_area - intersection
    return intersection / union if union > 0 else 0.0

# train classifer in val data
def train_classifier(val_imgs_path, val_masks_path, text_embeds, num_classes):
    X_train = []
    y_train = []
    
    for img_name in tqdm(os.listdir(val_imgs_path), desc="Collecting training data"):
        img = Image.open(os.path.join(val_imgs_path, img_name)).resize((224, 224))
        gt_mask = Image.open(os.path.join(val_masks_path, img_name.replace('.jpg', '.png')))
        gt_mask = np.array(gt_mask.resize((14, 14), Image.NEAREST)) // 255
        
        activations = get_clip_activations(img)  # [1, 196, 768]
        activations = activations.reshape(14, 14, -1).cpu().numpy()  # [14, 14, 768]
        
        for i in range(14):
            for j in range(14):
                X_train.append(activations[i, j])
                y_train.append(gt_mask[i, j])
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def generate_mask(img, text_embeds, num_classes, clf, threshold=0.5):
    img_shapes = np.array(img).shape[:2]
    img_resized = img.resize((224, 224))
    
    activations = get_clip_activations(img_resized)  # [1, 196, 768]
    activations = activations.reshape(14, 14, -1).cpu().numpy()  # [14, 14, 768]
    
    heatmap = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            prob = clf.predict_proba([activations[i, j]])[0, 1]
            heatmap[i, j] = prob
    
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().numpy()
    mask = (heatmap > threshold).astype(np.uint8)
    mask = binary_dilation(mask, iterations=1)
    
    mask = np.array(Image.fromarray(mask).resize(img_shapes[::-1], Image.NEAREST))
    return mask.astype(np.uint8), heatmap

def generate_seg_masks(imgs_path, val_masks_path=None, is_validation=False):
    text_embeds, num_classes = get_text_embeddings()
    img_names = []
    seg_masks = []
    
    clf = train_classifier(imgs_path if is_validation else val_imgs_path, val_masks_path, text_embeds, num_classes)
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_iou = 0.0
    if is_validation:
        temp_masks = []
        temp_names = []
        
        for img_name in tqdm(os.listdir(imgs_path), desc="Processing validation"):
            img = Image.open(os.path.join(imgs_path, img_name))
            mask, _ = generate_mask(img, text_embeds, num_classes, clf)
            temp_names.append(img_name)
            temp_masks.append(mask)
        
        for threshold in thresholds:
            ious = []
            for img_name, mask in zip(temp_names, temp_masks):
                gt_mask = Image.open(os.path.join(val_masks_path, img_name.replace('.jpg', '.png')))
                gt_mask = np.array(gt_mask) // 255
                pred_mask = (mask > threshold).astype(np.uint8)
                iou = binaryMaskIOU(pred_mask, gt_mask)
                ious.append(iou)
            
            mean_iou = np.mean(ious)
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_threshold = threshold
        
        print(f"Best threshold: {best_threshold}, Validation IoU: {best_iou}")
    
    for img_name in tqdm(os.listdir(imgs_path), desc="Processing images"):
        img = Image.open(os.path.join(imgs_path, img_name))
        mask, _ = generate_mask(img, text_embeds, num_classes, clf, best_threshold)
        img_names.append(img_name)
        seg_masks.append(mask)
    
    return img_names, seg_masks

def create_submission(img_names, seg_masks):
    ids = []
    b64 = []
    for img_name, seg_mask in zip(img_names, seg_masks):
        ids.append(img_name[:-4])
        seg_mask = (seg_mask * 255).astype(np.uint8)
        print(f"Processing {img_name}, mask shape: {seg_mask.shape}, dtype: {seg_mask.dtype}")
        mask = Image.fromarray(seg_mask, mode="L")
        buf = BytesIO()
        mask.save(buf, format="PNG")
        b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    
    df = pd.DataFrame({"img_id": [int(id_) for id_ in ids], "mask": b64})
    hsh = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    df.to_csv(submit_path, index=False)
    print(f"SUBMIT_NAME: {submit_path}")
    return df


val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'
val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, val_masks_path, is_validation=True)

val_ious = []
for img_name, seg_mask in zip(val_img_names, val_seg_masks):
    gt_mask = Image.open(os.path.join(val_masks_path, img_name.replace('.jpg', '.png')))
    gt_mask = np.array(gt_mask) // 255
    iou = binaryMaskIOU(seg_mask, gt_mask)
    val_ious.append(iou)
print(f"Mean Validation IoU: {np.mean(val_ious)}")

test_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/test_imgs'
test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, val_masks_path)
df = create_submission(test_img_names, test_seg_masks)










