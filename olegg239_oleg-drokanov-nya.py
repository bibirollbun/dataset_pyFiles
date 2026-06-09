class_names = [
    'american_bulldog',
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
    'cat',
    'dog',
    'Russian_Blue',
    'samoyed',
    'english_setter',
    'great_pyrenees'
]


animal_body_parts = [
    "whiskers",
    "tail",
    "paws",
    "claws",
    "fur",
    "ears",
    "nose",
    "eyes",
    "teeth",
    "tongue",
    "snout",
    "muzzle",
    "pads",
    "jaws",
    "legs",
    "spine",
    "belly",
    "back",
    "chest",
    "shoulders",
    "hips",
    "flanks",
    "neck",
    "ribcage",
    "forehead",
    "chin",
    "lip",
    "eyelid",
    "whisker pads",
    "dewclaw",
    "haunches",
    "hind legs",
    "forelegs",
    "elbow",
    "knee",
    "ankle",
    "thigh",
    "abdomen",
    "rump",
    "groin",
    "nape",
    "whisker roots",
    "scapula",
    "toes",
    "shoulder blades",
    "upper lip",
    "lower jaw",
    "bridge of nose",
    "eyebrows"
]

class_names += ["cat's " + x for x in animal_body_parts] + ["dog's " + x for x in animal_body_parts]


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


PATCH_SIZE = 64
CLIP_IMG_SIZE = 320
NUM_PATCHES = CLIP_IMG_SIZE // PATCH_SIZE * 2


def get_clip_representation(imgs):
    
    with torch.no_grad():
        inputs = processor(text='', images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        images_encoded = F.normalize(outputs.image_embeds, dim=-1)

    return images_encoded


background = items = [
    "tree", "bush", "flower", "bench", "fence", "lamp", "road", "car", "bicycle",
    "grass", "leaf", "stone", "path", "cloud", "sun", "bird", "butterfly", "squirrel",
    "pond", "lake", "river", "bridge", "hill", "mountain", "sky", "star", "moon",
    "house", "window", "door", "roof", "chimney", "porch", "balcony", "stair", "yard",
    "garden", "swing", "slide", "sandbox", "ball", "kite", "treehouse", "fireplace",
    "sofa", "chair", "table", "rug", "curtain", "shelf", "book", "lamp", "clock",
    "television", "picture", "mirror", "vase", "cushion", "blanket", "pillow", "basket",
    "plant", "cabinet", "drawer", "fridge", "stove", "oven", "microwave", "sink", "dish",
    "cup", "glass", "bottle", "fork", "knife", "spoon", "plate", "pan", "pot",
    "kettle", "toaster", "blender", "mug", "napkin", "tablecloth", "fireplace", "mantel",
    "rug", "floor", "ceiling", "wall", "painting", "frame", "fan", "heater", "air conditioner",
    "doorbell", "doormat", "umbrella", "boots", "coat", "hat", "scarf", "gloves", "leash",
    "collar", "bowl", "toy", "bed", "crate", "carrier", "brush", "shampoo", "towel",
    "basket", "laundry", "hanger", "ironing board", "iron", "vacuum", "broom", "mop", "bucket",
    "soap", "sponge", "cloth", "detergent", "cleaner", "duster", "dustpan", "trash can", "recycling bin",
    "mailbox", "newspaper", "magazine", "book", "journal", "pen", "pencil", "eraser", "sharpener",
    "notebook", "calendar", "clock", "watch", "glasses", "sunglasses", "phone", "charger", "tablet",
    "laptop", "computer", "keyboard", "mouse", "monitor", "printer", "camera", "tripod", "binoculars",
    "backpack", "bag", "purse", "wallet", "key", "lock", "alarm", "siren", "traffic light",
    "crosswalk", "stop sign", "signpost", "billboard", "poster", "graffiti", "streetlight", "pavement", "sidewalk"
]

background_tokenized = processor(text=background, images=None, return_tensors="pt", padding=True).to(device)
background_encoded = model.get_text_features(**background_tokenized)
background_encoded = F.normalize(background_encoded, dim=-1)


from skimage.measure import label, regionprops
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage.morphology import remove_small_holes, binary_erosion

def generate_seg_masks(imgs_path, classes_encoded, bg_encoded, batch_size=20):
    # Collect image paths
    img_paths = [os.path.join(imgs_path, img_name) for img_name in os.listdir(imgs_path)]
    imgs_names = []
    heatmaps = []

    # Process images in batches
    for i in tqdm(range(0, len(img_paths), batch_size)):
        batch_paths = img_paths[i:i + batch_size]
        batch_imgs = [Image.open(img_path) for img_path in batch_paths]

        # Save original sizes
        img_shapes = [np.array(img).shape for img in batch_imgs]

        # Get clip embeddings for the entire batch
        images_encoded = get_clip_representation(batch_imgs)

        # Determine the class for each image
        chosen_class_nums = (images_encoded @ classes_encoded.T).argmax(axis=1)
        chosen_class_embs = classes_encoded[chosen_class_nums]

        batch_heatmaps = []
        for img, chosen_class_emb in zip(batch_imgs, chosen_class_embs):
            # Resize and split image into patches
            img_resized = np.array(img.resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE)))
            img_patches = [
                Image.fromarray(img_resized[x:x + PATCH_SIZE, y:y + PATCH_SIZE])
                for x in range(0, CLIP_IMG_SIZE, PATCH_SIZE // 2)
                for y in range(0, CLIP_IMG_SIZE, PATCH_SIZE // 2)
            ]

            # Calculate cosine similarity between patch embeddings and class embedding
            img_patches_embs = get_clip_representation(img_patches)
            img_patches_embs_sims = img_patches_embs @ chosen_class_emb.unsqueeze(0).T
            
            img_patches_bg_sims = img_patches_embs @ bg_encoded.T
            thresholds, _ = img_patches_bg_sims.max(1)

            img_patches_embs_sims -= thresholds.unsqueeze(-1)

            # Form heatmap
            heatmap = img_patches_embs_sims
            heatmap = heatmap.reshape(NUM_PATCHES, NUM_PATCHES).unsqueeze(0)
            heatmap = torch.nn.functional.interpolate(
                heatmap[:, np.newaxis], scale_factor=PATCH_SIZE, mode='bilinear'
            ).to(device)

            # Normalize the heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            mean_heatmap_value = heatmap.mean()
            heatmap = heatmap.ge(mean_heatmap_value).type(heatmap.type())

            # Convert to numpy for region analysis
            heatmap_np = heatmap[0].cpu().detach().numpy().astype(bool)

            # Label connected regions
            labeled_heatmap = label(heatmap_np)
            regions = regionprops(labeled_heatmap)

            # Select the largest region
            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                mask = labeled_heatmap == largest_region.label

                # Fill holes in the largest region
                mask_filled = remove_small_holes(mask, area_threshold=100)

                batch_heatmaps.append(mask_filled)
            else:
                batch_heatmaps.append(np.zeros_like(heatmap_np))

        for img_name, heatmap, shape in zip(batch_paths, batch_heatmaps, img_shapes):
            target_transform = transforms.Compose([
                transforms.Resize((shape[0], shape[1]), Image.NEAREST),
            ])
            resized_heatmap = target_transform(torch.from_numpy(heatmap).unsqueeze(0)).squeeze(0)
            imgs_names.append(os.path.basename(img_name))
            heatmaps.append(resized_heatmap.squeeze().numpy())

    return imgs_names, heatmaps


val_imgs_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_imgs'
val_masks_path = '/kaggle/input/neoai-2025-cuties-segmentation/cuties/val_masks'

val_img_names, val_seg_masks = generate_seg_masks(val_imgs_path, classes_encoded, background_encoded)


import matplotlib.pyplot as plt
import os

NUM_VIS_IMG = 1
f, axes = plt.subplots(1,3)
axes[0].imshow(np.array(Image.open(os.path.join(val_imgs_path, 
                                                val_img_names[NUM_VIS_IMG]
                                               )
                                  ).resize((CLIP_IMG_SIZE, CLIP_IMG_SIZE))
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
test_img_names, test_seg_masks = generate_seg_masks(test_imgs_path, classes_encoded, background_encoded)


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
    mask = Image.fromarray(255*seg_mask.astype('float'))
    b64.append(image_to_base64(mask.convert("L")))

df = pd.DataFrame({"img_id": [int(id_) for id_ in ids], "mask": b64})
hsh = hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
submit_path = f"submit_{hsh}.csv"
print(f"SUBMIT_NAME: {submit_path}")
print(df.head(10))
df.to_csv(submit_path,index=False)

