import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
import torch
import cv2


pip install git+https://github.com/facebookresearch/segment-anything.git


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from skimage import io, transform
import torch.nn.functional as F

# Visualization functions
def show_mask(mask, ax, random_color=False):
    color = (
        np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        if random_color
        else np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    )
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024[None, None, :], dtype=torch.float, device=img_embed.device)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# Configurations
image_path = "/kaggle/input/demoimg/patient12_image_89.png"
input_box = np.array([45, 150, 175, 390])  # Input bounding box
checkpoint_path = "/kaggle/input/model-medsam/medsam_vit_b.pth"
device = "cuda"

# Load model
medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(device).eval()

# Load and preprocess image
img_np = io.imread(image_path)
img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
H, W, _ = img_3c.shape
img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
img_1024 = (img_1024 - img_1024.min()) / max(img_1024.max() - img_1024.min(), 1e-8)
img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

# Scale bounding box
box_1024 = input_box / np.array([W, H, W, H]) * 1024

# Perform inference
image_embedding = medsam_model.image_encoder(img_1024_tensor)
medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

# Visualize results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_3c)
show_box(input_box, ax[0])
ax[0].set_title("Input Image and Bounding Box")
ax[1].imshow(img_3c)
show_mask(medsam_seg, ax[1])
show_box(input_box, ax[1])
ax[1].set_title("MedSAM Segmentation")
plt.show()


