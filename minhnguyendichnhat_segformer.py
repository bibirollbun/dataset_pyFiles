from torch.utils.data import Dataset,DataLoader, random_split
from torchvision.transforms import Compose,Resize,PILToTensor,InterpolationMode,ToPILImage
import os
from PIL import Image
import time
import torch
import numpy as np
import torch.nn as nn
from torch.functional import F


import matplotlib.pyplot as plt
import wandb


input_path = '/kaggle/input/bkai-igh-neopolyp-small/train/train'
label_path = '/kaggle/input/bkai-igh-neopolyp-small/train_gt/train_gt'
test_path = '/kaggle/input/bkai-igh-neopolyp-small/test/test'


from albumentations import (
    Compose,
    RandomRotate90,
    Flip,
    HorizontalFlip,
    VerticalFlip,
    RandomGamma,
    RGBShift,
)



augmentation = Compose([
    RandomRotate90(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomGamma (gamma_limit=(70, 130), always_apply=False, p=0.2),
    RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
])


input_path = '/kaggle/input/bkai-igh-neopolyp/train/train'
label_path = '/kaggle/input/bkai-igh-neopolyp/train_gt/train_gt'
test_path = '/kaggle/input/bkai-igh-neopolyp/test/test'
input_list = os.listdir(input_path)
label_list = os.listdir(label_path)
test_list = os.listdir(test_path)
inputs_path = [input_path + '/' + i for i in input_list]
labels_path = [label_path + '/' + i for i in label_list]
tests_path = [test_path + '/' + i for i in test_list]


from transformers import SegformerImageProcessor

class SegDataClass(Dataset):
    def __init__(self, inputs_path,labels_path, transform, processor):
        super(SegDataClass, self).__init__()
        
        self.images_list = inputs_path
        self.masks_list = labels_path
        self.transform = transform
        self.processor = image_processor
    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]
        
        # Open image and mask
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = self.transform(mask) / 255
        
        mask = torch.where(mask > 0.65, 1.0, 0.0)
        mask[2, :, :] = 0.0001
        mask = np.array(torch.argmax(mask, 0)).astype(np.uint8)
        segmentation_map = Image.fromarray(mask)
        encoded_inputs = self.processor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        return encoded_inputs
    def __len__(self):
        return len(self.images_list)

transform = PILToTensor()
image_processor = SegformerImageProcessor(reduce_labels=False)

aug_dataset = SegDataClass(inputs_path, labels_path, transform=transform, processor=image_processor)


torch.manual_seed(42)


train_size = 0.9
valid_size = 0.1
batch_size = 4
train_aug_set, valid_aug_set = random_split(aug_dataset, 
                                    [int(train_size * len(aug_dataset)) , 
                                     int(valid_size * len(aug_dataset))])

train_dataloader = DataLoader(train_aug_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_aug_set, batch_size=batch_size, shuffle=True)


from transformers import SegformerForSemanticSegmentation
import json


id2label = {
    0: "neoplastic",
    1: "non-neoplastic",
    2: "background"
}
label2id = {v: k for k, v in id2label.items()}

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=3,
                                                         id2label=id2label,
                                                         label2id=label2id,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


!pip install evaluate


import evaluate

metric = evaluate.load("mean_iou")


import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create directory for saving the best model
save_dir = "best_model"
os.makedirs(save_dir, exist_ok=True)

# Initialize the best metric for tracking
best_mean_iou = float('-inf')  # Start with a very low value

model.train()
for epoch in range(25):  # loop over the dataset multiple times
    print(f"Epoch: {epoch}")
    
    # Training phase
    for idx, batch in enumerate(tqdm(train_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # Evaluate within training loop
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # Note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # Print loss and metrics every 100 batches
        if idx % 100 == 0:
            # Currently using _compute instead of compute
            metrics = metric._compute(
                predictions=predicted.cpu(),
                references=labels.cpu(),
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=False,  # We've already reduced the labels ourselves
            )

            print(f"Loss: {loss.item()}")
            print(f"Mean_iou: {metrics['mean_iou']}")
            print(f"Mean accuracy: {metrics['mean_accuracy']}")

    # Evaluate on validation dataset after each epoch
    model.eval()  # Set model to evaluation mode
    val_metrics = {"mean_iou": 0, "mean_accuracy": 0}
    val_batches = len(valid_dataloader)
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits

            # Upsample and get predictions
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # Add to metric for validation
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # Compute the final metrics for validation
        val_metrics = metric._compute(
            predictions=predicted.cpu(),
            references=labels.cpu(),
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=False,
        )

    print(f"Validation Mean IoU: {val_metrics['mean_iou']}")
    print(f"Validation Mean Accuracy: {val_metrics['mean_accuracy']}")

    # Check if this epoch's validation metrics are the best
    if val_metrics["mean_iou"] > best_mean_iou:
        best_mean_iou = val_metrics["mean_iou"]
        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))
        print("Best model saved!")

    model.train()  # Switch back to training mode



model.eval()
from tqdm import tqdm
if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
model = model.to(device)
for path in tqdm(tests_path):
    image = Image.open(tests_path[0])
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    image_id = path.split('/')[-1].split('.')[0]
    filename = image_id + ".png"
    mask2img = ToPILImage()(F.one_hot(torch.tensor(predicted_segmentation_map)).permute(2, 0, 1).float())
    mask2img.save(os.path.join("/kaggle/working/predicted_masks", filename))


import cv2
import pandas as pd
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r

MASK_DIR_PATH = '/kaggle/working/predicted_masks' # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output_v1.csv', index=False)





