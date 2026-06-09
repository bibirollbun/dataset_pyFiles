import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


# 1. Paths
val_path = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
labels_path = "/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv"
mapping_path = "/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"


import torch
from torchvision.models import alexnet

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = alexnet(weights="DEFAULT").to(device)  ## ...
model.eval()



import os
import random
import torch
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load pretrained AlexNet
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = alexnet(weights="DEFAULT").to(device)
# model.eval()

# 2. Load val image paths
val_path = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
sample_imgs = random.sample(sorted(os.listdir(val_path)), 20)

# 3. Load synset-to-name mapping
label_map_path = "/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt"
synset_to_label = {}
with open(label_map_path, "r") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            synset, label = parts
            synset_to_label[synset] = label.split(",")[0]

# 4. Load ground truth labels
df_labels = pd.read_csv("/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv")
df_labels.columns = ['ImageId', 'Label']
df_labels['Label'] = df_labels['Label'].str.split().str[0]

# 5. ImageNet class index → label mapping
idx_to_label = AlexNet_Weights.IMAGENET1K_V1.meta["categories"]

# 6. Define transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 7. Show 20 random images with predictions
rows, cols = 4, 5
plt.figure(figsize=(20, 15))

for i, fname in enumerate(sample_imgs):
    img_id = fname.split(".")[0]
    img_path = os.path.join(val_path, fname)
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

    # Predicted label
    predicted = idx_to_label[pred_class]

    # True label
    true_synset = df_labels[df_labels["ImageId"] == img_id]["Label"].values[0]
    true_label = synset_to_label.get(true_synset, true_synset)

    # Plot
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Pred: {predicted}\nTrue: {true_label}", fontsize=12)

plt.tight_layout()
plt.show()


for name, p in model.named_parameters():
    if name.endswith(".weight") and "features" in name:
        print(name, tuple(p.shape))



model.features[0]


model.features[0].weight.size()


model.features[0].weight.shape   # (64, 3, 11, 11)


model.features[0].out_channels   # 64


model.features[0].bias.shape


model.features[1]


model.features[3].weight.size()


model.features[6].weight.size()


model.features[8].weight.size()


model.features[10].weight.size()


model.features[0].kernel_size         # -> (11, 11)
model.features[3].kernel_size         # -> (5, 5)
model.features[6].kernel_size         # -> (3, 3)
model.features[8].kernel_size         # -> (3, 3)
model.features[10].kernel_size        # -> (3, 3)


params_list = list(model.parameters())
len(params_list)      



total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  |  Trainable: {trainable:,}")


import torch
import matplotlib.pyplot as plt
from torchvision.models import alexnet, AlexNet_Weights

# Load pretrained AlexNet
weights = AlexNet_Weights.DEFAULT
model = alexnet(weights=weights).eval()

# Grab the first conv layer
conv1 = model.features[0]

# Extract weights: shape [64, 3, 11, 11]
kernels = conv1.weight.data.clone()

# Normalize each kernel for visualization
def normalize_kernel(k):
    k = k - k.min()
    k = k / k.max()
    return k


# Plot all 64 kernels
fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    kernel = normalize_kernel(kernels[i])
    # Move from [3,11,11] -> [11,11,3] for RGB plotting
    kernel = kernel.permute(1, 2, 0).numpy()
    ax.imshow(kernel)
    ax.axis("off")
plt.suptitle("All 64 Conv1 Kernels (AlexNet)")
plt.show()



import torch, torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# --- utils ---
def save_grid(chw_list, nrow, path, norm=True, scale_each=True):
    tiles = []
    for t in chw_list:
        if t.ndim == 2: t = t[None, ...]      # (H,W) -> (1,H,W)
        if t.shape[0] == 1: t = t.repeat(3,1,1)  # grayscale -> RGB for viewing
        tiles.append(t.cpu())
    grid = make_grid(torch.stack(tiles,0), nrow=nrow, normalize=norm, scale_each=scale_each, padding=2)
    plt.figure(figsize=(12,12)); plt.axis('off'); plt.imshow(grid.permute(1,2,0))
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches='tight'); plt.close()
    print("saved:", path)

def full_corr2d(a, b):  # “full” cross-correlation of 2D kernels (for composing kernels)
    ph, pw = b.shape[-2]-1, b.shape[-1]-1
    a = F.pad(a[None,None], (pw,pw,ph,ph), mode='constant', value=0)  # 1x1xH'×W'
    b = b[None,None]
    y = F.conv2d(a, b)  # 1x1x(Ha+Hb-1)x(Wa+Wb-1)
    return y[0,0]

# --- 1) Conv1 filters as RGB tiles ---
W1 = model.features[0].weight.data.clone()          # (64, 3, 11, 11) in torchvision AlexNet
# sort by L2 norm just to put chunky filters first
idx1 = torch.topk(W1.view(W1.size(0), -1).norm(dim=1), k=W1.size(0)).indices
save_grid([W1[i] for i in idx1], nrow=8, path="conv1_rgb_filters.png")

# --- 2) Conv2 filters projected to pixel space via Conv1 (≈ 11×11 RGB) ---
W2 = model.features[3].weight.data.clone()          # (192, 64, 5, 5)
k_eff = W1.size(-1) + W2.size(-1) - 1               # 11 + 5 - 1 = 15 for AlexNet
W2_px = torch.zeros(W2.size(0), 3, k_eff, k_eff)    # (192,3,15,15)

# compose: for each Conv2 filter m and RGB channel c, sum over Conv1 ch k of (W1[k,c] (*) W2[m,k])
with torch.no_grad():
    for m in range(W2.size(0)):
        for c in range(3):
            acc = torch.zeros(k_eff, k_eff)
            for k in range(W1.size(0)):
                acc += full_corr2d(W1[k, c], W2[m, k])
            W2_px[m, c] = acc

idx2 = torch.topk(W2.view(W2.size(0), -1).norm(dim=1), k=min(64, W2.size(0))).indices
save_grid([W2_px[i] for i in idx2], nrow=8, path="conv2_pixelspace_top64.png")

# --- 3) Channel-slice heatmaps for any deeper conv (e.g., Conv3) ---
def visualize_conv_channel_slices(conv_module, out_index=0, top_slices=64, fname="convX_filter_slices.png"):
    W = conv_module.weight.data.clone()             # (out_ch, in_ch, kH, kW)
    m = int(torch.topk(W.view(W.size(0), -1).norm(dim=1), k=1).indices) if out_index is None else out_index
    Wm = W[m]                                       # (in_ch, kH, kW)
    # rank input-channel slices by energy
    idx = torch.topk(Wm.view(Wm.size(0), -1).norm(dim=1), k=min(top_slices, Wm.size(0))).indices
    save_grid([Wm[i] for i in idx], nrow=16, path=fname)

# examples:
visualize_conv_channel_slices(model.features[6], out_index=None, top_slices=64, fname="conv3_filter_slices_top64.png")
visualize_conv_channel_slices(model.features[8], out_index=None, top_slices=64, fname="conv4_filter_slices_top64.png")
visualize_conv_channel_slices(model.features[10], out_index=None, top_slices=64, fname="conv5_filter_slices_top64.png")



from IPython.display import Image, display
display(Image(filename="/kaggle/working/conv1_rgb_filters.png"))


display(Image(filename="/kaggle/working/conv2_pixelspace_top64.png"))


display(Image(filename="/kaggle/working/conv3_filter_slices_top64.png"))


from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as IPyImage, display

paths = [
    "/kaggle/working/conv3_filter_slices_top64.png",
    "/kaggle/working/conv4_filter_slices_top64.png",
    "/kaggle/working/conv5_filter_slices_top64.png",
]
labels = ["Conv3 slices", "Conv4 slices", "Conv5 slices"]

# ----- sizing & layout -----
target_w = 900   # <- make wider/narrower here
gutter   = 20    # vertical space between panels
label_h  = 40    # label bar height
side_pad = 20    # left/right padding

# Pillow 10+ resampling compatibility
Resampling = getattr(Image, "Resampling", Image)

# load & resize (keep aspect)
imgs = [Image.open(p).convert("RGB") for p in paths]
resized = []
for im in imgs:
    w, h = im.size
    scale = target_w / max(1, w)
    nh = int(round(h * scale))
    resized.append(im.resize((target_w, nh), Resampling.BILINEAR))

# canvas size
total_w = target_w + side_pad * 2
total_h = sum(im.size[1] for im in resized) + (label_h * len(resized)) + gutter * (len(resized) - 1)
canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except Exception:
    font = ImageFont.load_default()

def text_wh(draw_obj, text, font_obj):
    if hasattr(draw_obj, "textbbox"):  # Pillow ≥10
        l, t, r, b = draw_obj.textbbox((0, 0), text, font=font_obj)
        return r - l, b - t
    if hasattr(font_obj, "getsize"):   # fallback
        return font_obj.getsize(text)
    w = draw_obj.textlength(text, font=font_obj)
    ascent, descent = font_obj.getmetrics()
    return int(w), int(ascent + descent)

# paste vertically with labels
y = 0
for im, lab in zip(resized, labels):
    # image
    canvas.paste(im, (side_pad, y))
    y += im.size[1]

    # label centered under panel
    tw, th = text_wh(draw, lab, font)
    cx = side_pad + im.size[0] // 2
    draw.text((cx - tw // 2, y + (label_h - th) // 2), lab, fill=(30, 30, 30), font=font)

    y += label_h + gutter  # move down for next panel

out_path = "/kaggle/working/conv3to5_slices_stack.png"
canvas.save(out_path, quality=95)
display(IPyImage(filename=out_path))
print("Saved:", out_path)


