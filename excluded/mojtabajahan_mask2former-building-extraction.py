# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# os.system("wget https://raw.githubusercontent.com/pytorch/vision/refs/heads/main/gallery/transforms/helpers.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")


!pip install -q git+https://github.com/huggingface/transformers.git


import os
import torch
import json
import utils
import cv2
import torchvision
import random

import numpy as np

from torchvision import transforms as _transforms
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform, RandomZoomOut
from torchvision.transforms import v2 as T

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation


# Rotating images and masks trasform
class Rotate(object):
    def __call__(self, image, target):        
        rnd = random.randint(0,3)
        degree = rnd * 90
        if (degree != 0):
            image = F.rotate(image, degree)
            rotated_masks = F.rotate(target, degree)            
            return image, rotated_masks
        else:
            return image, target
            
class Flip(object):
    def __call__(self, image, target):
        # flipped_image = none
        flipped_masks = None
        rnd = random.random()
        if(rnd >= 0.5):
            if (rnd >= 0.75):
                flipped_image = F.hflip(image)
                flipped_masks = F.hflip(target)
            else:
                flipped_image = F.vflip(image)
                flipped_masks = F.vflip(target)                
            return flipped_image, flipped_masks
        else:
            return image, target
        
class Zoomout(object):
    def __call__(self, image, target):
        zoom_out = RandomZoomOut(side_range=(1.2, 1.4), fill=0.5, p=0.5)
        out = zoom_out({"input": image, "target": target["masks"]})                
        target["boxes"] = tv_tensors.BoundingBoxes(masks_to_boxes(out["target"]), format="XYXY", canvas_size=F.get_size(image))
        target["masks"] = out["target"]
        return out["input"], target
   

# Decreasing image quality transform
class DecreaseQuality(object):
    
    def __call__(self, image, target):     
        if(random.random() >= 0.5):
            width = image.shape[1]
            height = image.shape[2]
            decrease_size = T.Resize((int(width/2), int(height/2)))  # Specify desired size
            increase_size = T.Resize((width, height))
            resized_image = decrease_size(image)        
            decreased_quality_img = increase_size(resized_image)
            return decreased_quality_img, target
        else:
            return image, target

class Resize:    
    prev_scale = None
    def __init__(self, size):
        self.size = size  # e.g. (height, width)

    def __call__(self, img, masks):  
        if (self.size == None):
            scales = [512, 650, 800]
            scale = random.choice(scales)
            scale = (scale, scale) 
        else:
            scale = self.size
        # print("Prev Scale:", Resize.prev_scale)        
        # Resize the image (bilinear interpolation)        
        img = F.resize(img, scale, interpolation=F.InterpolationMode.BILINEAR)

        # Resize each mask (nearest-neighbor to avoid smoothing labels)
        resized_masks = []
        for m in masks:
            m = F.resize(m.unsqueeze(0), scale, interpolation=F.InterpolationMode.NEAREST)
            resized_masks.append(m.squeeze(0))

        # Stack into a single tensor (N, H, W)
        resized_masks = torch.stack(resized_masks, dim=0)

        return img, resized_masks


import torchvision.transforms.functional as F

IMAGE_SIZE = (512, 512)

# Dataset Helper Functions
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
    
def get_image_name(image_path):
    splited = image_path.split("/")
    return "/".join(splited[-2:])

def get_image_annotations(json_data, image_name):
    image_dic = None
    annot_list = []
    for image in json_data['images']:
        if (image['file_name'] == image_name):
            image_dic = image
            break
    found = False
    for annot in json_data['annotations']:
        if (annot['image_id'] == image_dic['id']):
            annot_list.append(annot)
            found = True
        if (found and annot['image_id'] != image_dic['id']):            
            break
    
    # Convert the list of annotations dictionary to a dictionary of annotaions list
    annot_dic = {key: [d[key][0] if(key == 'segmentation') else d[key] for d in annot_list] for key in annot_list[0].keys()}    
    annotation = Annotations(annot_dic, image_dic)
    return annotation

def create_pairs(input_list):
  """Converts a list to a set of pairs.

  Args:
    input_list: The input list.

  Returns:
    A set of pairs.
  """
  pairs = []
  for i in range(0, len(input_list) - 1, 2):
    pairs.append(tuple(input_list[i:i+2]))
  return pairs

def create_contour_mask(contour_points, image_shape):
    """
    Create a binary mask from contour points.

    Args:
        contour_points: List of tuples [(x,y), (x1,y1), ..., (xn,yn)]
        image_shape: Tuple of (height, width) for the output mask

    Returns:
        Binary mask as numpy array where filled contour is 1, rest is 0
    """
    # Convert list of tuples to numpy array and reshape for cv2    
    contour = np.array(contour_points).reshape((-1,1,2)).astype(np.int32)

    # Create empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Fill the contour
    cv2.fillPoly(mask, [contour], color=1)

    return torch.tensor(mask)

def create_masks_from_contours(point_list, image_width, image_height):
    masks = []
    image_shape = (image_height, image_width)
    for seg in point_list:        
        points_pair = create_pairs(seg)  
        if (len(points_pair) > 3):
            masks.append(create_contour_mask(points_pair, image_shape))            
    if (len(masks) == 0):
        return None
    return torch.stack(masks, dim=0)

def clean_boxes_masks(boxes, masks):        
    removed_indices = []
    for i, box in enumerate(boxes):
        if( box[0] == box[2] or box[1] == box[3]):
            removed_indices.append(i)
            
    if (len(removed_indices) > 0):
        filtered_numbers = [num for num in range(0, masks.shape[0]) if num not in removed_indices]
        boxes = boxes[filtered_numbers]
        masks = masks[filtered_numbers]

    return boxes, masks

def get_transform(train):
    transforms = []
    if train:        
        # transforms.append(DecreaseQuality())        
        transforms.append(Rotate())
        transforms.append(Flip())
        # transforms.append(Zoomout())    
        transforms.append(Resize(None))
    else:
        transforms.append(Resize(IMAGE_SIZE))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


# Annotations Class
class Annotations():
    def __init__(self, annotation_dic, image_dic):
        self.image_id = image_dic['id']
        self.image_height = image_dic['height']
        self.image_width = image_dic['width']
        self.bbox_list = annotation_dic['bbox']
        self.segmentation_list = annotation_dic['segmentation']
        self.area_list = annotation_dic['area']    


'''
Replace all 1 values with their corresponding mask object IDs and merge for Mask2Former preprocessing
Args:
    masks (torch.Tensor): A 3D tensor of shape (N, H, W)
        - N = number of masks (instances or categories).
        - Each slice masks[i] is a 2D mask of shape (H, W), where:
            * 0 = background
            * 1 (or nonzero) = pixels belonging to that instance.
Returns:
    merged (torch.Tensor): A 2D tensor of shape (H, W)
        - Contains integer labels:
            * 0 = background
            * 1 = pixels from first mask
            * 2 = pixels from second mask
            * ...
            * N = pixels from N-th mask
        - If multiple masks overlap, the later mask in the stack
          (higher index i) overwrites earlier ones.
'''
def mergeMasks(masks):
    merged = torch.zeros_like(masks[0])
    for i in range(masks.shape[0]):
        msk = torch.where(masks[i] == 1, i+1, masks[i])
        merged = torch.where(masks[i] != 0, msk, merged)
    return merged


# Create Dataset
class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.annot_json = read_json_file(os.path.join(root, annotation_file))

    def __getitem__(self, idx):
        # load images and masks        
        img_path = os.path.join(self.root, "image", self.imgs[idx])        
        image_name = get_image_name(img_path)        
        img = read_image(img_path)                
        
        # get image annotations
        annotations = get_image_annotations(self.annot_json, image_name)                     
        # num_objs = len(annotations.bbox_list)

        # generate masks from contours          
        H, W = img.shape[1], img.shape[2]
        masks = create_masks_from_contours(annotations.segmentation_list, W, H)        

        if(masks == None):            
            return None
        
        if self.transforms is not None:            
            img, masks = self.transforms(img, masks)
            
        merged_masks = mergeMasks(masks)
        # print("merged_masks unique:", torch.unique(merged_masks))
        inst2class = {i+1: 0 for i in range(masks.shape[0])}        
        # print("inst2class", inst2class)

        inputs = processor([img], [merged_masks], instance_id_to_semantic_id=inst2class, return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
        inputs["image_id"] = annotations.image_id
        inputs["height"] = annotations.image_height
        inputs["width"] = annotations.image_width
        return inputs

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    valid_examples = [example for example in batch if example is not None]
    if (len(valid_examples) > 0):
        pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])
        pixel_mask = torch.stack([example["pixel_mask"] for example in valid_examples])
        class_labels = [example["class_labels"] for example in valid_examples]
        mask_labels = [example["mask_labels"] for example in valid_examples]    
        image_ids = [example["image_id"] for example in valid_examples]
        image_heights = [example["height"] for example in valid_examples]
        image_widths = [example["width"] for example in valid_examples]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels, "image_ids": image_ids, "image_heights": image_heights, "image_widths": image_widths}
    else:
        return None


path = '/kaggle/input/building-extraction-generalization-2024'
train_dataset = BuildingDataset(f'{path}/train', 'train.json', get_transform(train=True))
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = BuildingDataset(f'{path}/val', 'val.json', get_transform(train=False))
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)


# Show samples from dataset
import matplotlib.pyplot as plt
from helpers import plot
from torchvision.tv_tensors import Image
from torchvision.utils import draw_bounding_boxes

def display_tensor_image(tensor_image, title=None):
    """
    Display a PyTorch tensor image that can handle both grayscale and RGB formats
    
    Args:
        tensor_image (torch.Tensor): Image tensor with shape:
            - [H, W] for grayscale images
            - [3, H, W] for RGB images (channel-first format)
        title (str, optional): Custom title for the plot
    """
    # Convert to numpy and move to CPU if needed
    image_np = tensor_image.detach().cpu().numpy()
    
    # Handle different tensor shapes
    if len(image_np.shape) == 2:  # Grayscale: [H, W]
        H, W = image_np.shape
        cmap = 'gray'
        shape_info = f"Grayscale [{H}, {W}]"
        
    elif len(image_np.shape) == 3 and image_np.shape[0] == 3:  # RGB: [3, H, W]
        C, H, W = image_np.shape
        # Convert from [C, H, W] to [H, W, C] for matplotlib
        image_np = image_np.transpose(1, 2, 0)
        cmap = None
        shape_info = f"RGB [3, {H}, {W}]"
        
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor_image.shape}. "
                        f"Expected [H, W] for grayscale or [3, H, W] for RGB")
    
    # Normalize values if needed
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    
    # Clip values to ensure they're in valid range [0, 1]
    image_np = np.clip(image_np, 0, 1)
    
    # Create the plot
    plt.figure(figsize=(5, 5))
    plt.imshow(image_np, cmap=cmap)
    plt.axis('off')
    
    # Set title
    if title:
        plt.title(f'{title}\nShape: {shape_info}')
    else:
        plt.title(f'PyTorch Tensor Image\nShape: {shape_info}')
    
    plt.tight_layout()
    plt.show()



'''inputs = dataset[0]
for k,v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(k,v.shape)
    if k == "pixel_mask":
        print(v)
    if k == "mask_labels":
        print(display_tensor_image(v[1]))
   '''     
# batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   if isinstance(v, torch.Tensor):
#     print(k,v.shape)
#   else:
#     print(k,len(v)) 

# print(batch['mask_labels'][0].shape)
# display_tensor_image(batch['pixel_values'][0])
# display_tensor_image(batch['mask_labels'][0][0])


import torch
from tqdm.auto import tqdm
from transformers import MaskFormerForInstanceSegmentation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from transformers import Mask2FormerImageProcessor
from transformers import get_scheduler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as FN

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

processor = Mask2FormerImageProcessor(reduce_labels=False, ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)
num_epochs = 7

def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    else:
        return batch

def setBackboneUnLearnable(model, learnable):
    for name, param in model.named_parameters():
        if (name.startswith('model.pixel_level_module.encoder')):
            param.requires_grad = learnable

def convert_to_coco_predictions(outputs, image_ids, image_widths, image_heights, processor, threshold=0.5):
    """
    Convert batched Mask2Former outputs into COCO-style predictions.

    Args:
        outputs: model output from Mask2Former (batched)
        image_ids: list of image IDs (length = batch size)
        processor: (not used here but kept for consistency)
        threshold: score threshold
        target_size: (H, W) for resizing masks to original or fixed size
    """
    pred_logits = outputs.class_queries_logits     # [B, num_queries, num_classes+1]
    pred_masks = outputs.masks_queries_logits       # [B, num_queries, H, W]

    batch_size = pred_logits.shape[0]
    coco_results = []

    for b in range(batch_size):
        image_id = image_ids[b]
        logits = pred_logits[b]   # [num_queries, num_classes+1]
        masks = pred_masks[b]     # [num_queries, H, W]
        image_height = image_heights[b]
        image_width = image_widths[b]
        target_size = (image_width, image_height)

        # Compute probabilities and labels (remove "no object" class)
        probs = logits.softmax(-1)[:, :-1]
        scores, labels = probs.max(-1)

        # Resize and threshold masks
        masks = masks.unsqueeze(1)  # [num_queries, 1, H, W]
        masks = FN.interpolate(masks, size=target_size, mode="bilinear", align_corners=False)
        masks = masks.squeeze(1).sigmoid().cpu().numpy() > 0.5

        for score, label, mask in zip(scores, labels, masks):
            if score < threshold:
                continue

            # RLE encoding for COCO
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(label.item()) if hasattr(label, "item") else int(label),
                "segmentation": rle,
                "score": float(score.item())
            })

    return coco_results

def evaluate(model, dataloader, device="cuda"):
    model.eval()
    coco_gt = COCO("/kaggle/input/building-extraction-generalization-2024/val/val.json")
    
    all_coco_results = []    
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        # pixel_values = batch["pixel_values"][1::].to(device)
        # pixel_mask = batch["pixel_mask"][1::].to(device)
        # image_id = batch["image_ids"][1]
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        image_ids = batch["image_ids"]    
        image_widths = batch["image_widths"]
        image_heights = batch["image_heights"]
        
        # mask_labels = mergeMasks(batch["mask_labels"][1].to(device)).unsqueeze(0)            
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)                
    
        preds = convert_to_coco_predictions(outputs, image_ids, image_widths, image_heights, processor)        
        all_coco_results += preds
    # print(all_coco_results)
    coco_gt.dataset['info'] = {"info":{"year": 2025, "version": "1.0", "description": "building", "contributor": "empty", "url": "empty", "date_created": 20250810}}
    # print(all_coco_results)
    coco_dt = coco_gt.loadRes(all_coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def save_last_steps(last_steps):
    print("Last Steps:", last_steps)
    with open("last_steps.txt", "w") as f:
        f.write(str(last_steps))

def load_last_steps():
    try:
        with open("last_steps.txt", "r") as f:
            return int(f.read())
    except FileNotFoundError:
        return 0

def create_scheduler(optimizer, train_dataloader):    
    lr_scheduler = None
    num_training_steps = num_epochs * len(train_dataloader)
    last_steps = max([0, load_last_steps() - 6000])
    STEP_SIZE = 500
    type = "PyTorch"
    if (type == "transformer"):
        lr_scheduler = get_scheduler(
        name="linear",         # or "cosine", "polynomial", "constant", etc.
        optimizer=optimizer,
        num_warmup_steps=5,  # typical: 500–1000 for transformer fine-tuning
        num_training_steps=num_training_steps,)
    elif (type == "PyTorch"):        
        print("LR:", optimizer.param_groups[0]["lr"])
        print("last_steps:", last_steps)
        lr = optimizer.param_groups[0]['lr']
        gamma = 0.95        
        lr = (gamma ** int(last_steps / STEP_SIZE)) * lr        
        print("LR:", f"{lr:.7f}")
        lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.95)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.lr = lr

    return lr_scheduler

try:
    model = Mask2FormerForUniversalSegmentation.from_pretrained("mask2former-checkpoint")
except Exception as e:
    if(str(e).startswith("mask2former-checkpoint is not a local folder")):
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
setBackboneUnLearnable(model, True)

backbone_params = [p for n,p in model.named_parameters() if "encoder" in n]
head_params = [p for n,p in model.named_parameters() if "encoder" not in n]

optimizer = torch.optim.Adam([
    {"params": head_params,     "lr": 2e-4},
    {"params": backbone_params, "lr": 1e-4}    
], lr=2e-4)

lr_scheduler = create_scheduler(optimizer, train_dataloader)

last_steps = load_last_steps()
steps = 0
for epoch in range(num_epochs):
  running_loss = 0.0
  num_samples = 0
  print("Epoch:", epoch)
  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
      # break
      # Reset the parameter gradients
      optimizer.zero_grad()
      if(batch == None):
          continue
      # Forward pass
      outputs = model(
              pixel_values=batch["pixel_values"].to(device),
              mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
              class_labels=[labels.to(device) for labels in batch["class_labels"]],
      )

      # Backward propagation
      loss = outputs.loss
      loss.backward()

      batch_size = batch["pixel_values"].size(0)
      running_loss += loss.item()
      num_samples += batch_size

      

      if idx % 100 == 0:
          print("Loss:", running_loss/num_samples)   
          
          lr_sch = lr_scheduler.get_last_lr()[0]    
          l = optimizer.param_groups[0]['lr']
          print("LR rate:" ,f"{l:.7f}")
          print("LR scheduler:", f"{lr_sch:.7f}")
          model.save_pretrained("mask2former-checkpoint")
          steps += 100
          save_last_steps(steps + last_steps)

      # Optimization
      optimizer.step()
      if (last_steps > 0 or epoch > 0):
          lr_scheduler.step()      
      optimizer.zero_grad()
      
  evaluate(model, val_dataloader, device=device)


!rm -r mask2former-checkpoint
!rm last_steps.txt


import matplotlib.pyplot as plt
import torch.nn.functional as FN
from PIL import Image
import pandas as pd
from transformers import Mask2FormerImageProcessor, AutoImageProcessor

# processor = Mask2FormerImageProcessor(reduce_labels=False, ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("mask2former-checkpoint")
model.eval()

def convert_mask_to_polygon(mask):
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    coords = []        
    for obj in contours:        
        for point in obj:
            coords.append((int(point[0][0]),int(point[0][1])))
    return coords

def create_polygon_from_contours(mask):    
    threshold = 0.5
    mask = mask.squeeze(0)
    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0
    mask = mask.to(dtype=torch.uint8)
    numpy_mask = mask.detach().cpu().numpy()    
    return convert_mask_to_polygon(numpy_mask)

def generate_submission(image_list):
    df = pd.DataFrame(image_list)
    df.rename(columns={"imageId": "ImageID", "objectList": "Coordinates"}, inplace=True)                                                            
    df.to_csv("submission.csv", index=False, header=True)
    
test_data_path = '/kaggle/input/building-extraction-generalization-2024/test'
image_file_list = list(sorted(os.listdir(os.path.join(test_data_path, "image"))))
image_polygon_list = []
score_threshold = 0.5

for image_id, image_file in enumerate(image_file_list): 
    print("ImageID:", image_id)
    pil_image = Image.open(os.path.join(test_data_path, "image", image_file))    
    inputs = image_processor(pil_image, return_tensors="pt")
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].float()
    if 'pixel_mask' in inputs:
        inputs['pixel_mask'] = inputs['pixel_mask'].float()

    outputs = model(**inputs)
    mask_logits  = outputs["masks_queries_logits"] 
    class_logits = outputs["class_queries_logits"]
    
    masks = FN.interpolate(mask_logits, size=pil_image.size, mode="bilinear", align_corners=False)
    print(masks.shape)
    probs = class_logits.softmax(-1)[0, :, :-1]  # ignore "no object" class
    scores, labels = probs.max(-1)
    polygon_list = []
    for index, mask in enumerate(masks[0]):             
        mask_probs = torch.sigmoid(mask)
        if (scores[index] >= score_threshold):
            polygon = create_polygon_from_contours(mask_probs)
            if (len(polygon) > 3): 
                polygon_list.append(polygon)
        # plt.imshow(mask_probs.cpu().detach(), cmap='gray')
        # plt.title("Predicted Mask")
        # plt.colorbar()
        # plt.show()
        # if(index == 2):
        #     break
    image_polygon_list.append({'imageId': image_id, 'objectList': str(polygon_list)})
    print("LEN:", len(image_polygon_list))    
    
generate_submission(image_polygon_list)


!echo '{"username":"mojtabajahan","key":"f32732ef8da53c045f87de56c8e1d5a5"}'

