import os 
import cv2 
import random 
import zipfile
from tqdm import tqdm

# Pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Augmentation 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

# Data Processing 
import numpy as np 
import pandas as pd
from PIL import Image

# Visualization 
import matplotlib.pyplot as plt


class EncoderResnet34(nn.Module):
    def __init__(self, backbone='resnet34', use_pretrained=True):
        super(EncoderResnet34, self).__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None)
        self.encoder = nn.ModuleList([
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu), # 64 - H/2
            self.resnet.maxpool, # H/4
            self.resnet.layer1, # 64 - H/4
            self.resnet.layer2, # 128 - H/8
            self.resnet.layer3, # 256 - H/16
        ])
        # take the last layer of Resnet as the bottleneck
        self.bottleneck = self.resnet.layer4 # 512 - H/32

    def forward(self, x):
        skip_connections = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != 1:
                skip_connections.append(x)

        x = self.bottleneck(x)
            
        return x, skip_connections   


class DecoderModule(nn.Module):
    # This Decoder Module include conv block + up-sample(conv transpose)
    def __init__(self, in_channels, out_channels):
        super(DecoderModule, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2 , stride = 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)    
        )
        
    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        
        if skip_connection.shape[2:] != x.shape[2:]: 
                x = TF.resize(x, size=skip_connection.shape[2:]) 
            
        x_concat = torch.cat([skip_connection, x], dim=1)
        return self.conv(x_concat)


class Decoder(nn.Module):
    def __init__(self, features=[512, 256, 128, 64, 64]):
        super(Decoder, self).__init__()
        self.features = features 
        self.decoder = nn.ModuleList()

        for i in range(len(self.features)-1):
            in_channels = self.features[i]
            out_channels = self.features[i+1]
            self.decoder.append(DecoderModule(in_channels, out_channels))
        
    def forward(self, x, skip_connections):
        # reverse the skip connection list
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.decoder):
            skip_connection = skip_connections[i]
            x = layer(x, skip_connection)
        return x


class UNetResnet34(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[512, 256, 128, 64, 64]):
        super(UNetResnet34, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features 

        self.encoder = EncoderResnet34()
        self.decoder = Decoder(self.features)

        last_channel = features[-1]
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(last_channel, last_channel, kernel_size=2 , stride = 2),
            
            nn.Conv2d(last_channel, last_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_channel, last_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(last_channel, self.out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        return self.final_conv(x)
    


def test():
    x = torch.randn((3, 3, 512, 512))
    model = UNetResnet34(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    assert preds.shape[2:] == x.shape[2:]

test()

x = torch.randn((3, 3, 161, 161))



# Unziping data files 
data_path = "/kaggle/input/carvana-image-masking-challenge/"

def unzip(zip_file_name, extract_dest):
    with zipfile.ZipFile(os.path.join(data_path, zip_file_name), 'r') as z:
        z.extractall(extract_dest)
        print(f"Extracted {zip_file_name} -> {extract_dest}")


rm -rf /kaggle/working/*


unzip("test_hq.zip", "/kaggle/working/")
unzip("train_hq.zip", "/kaggle/working/")
unzip("train_masks.zip", "/kaggle/working/")


print(os.listdir("/kaggle/working/train_hq")[:5])
print(os.listdir("/kaggle/working/train_masks")[:5])

print(len(os.listdir("/kaggle/working/train_hq")))
print(len(os.listdir("/kaggle/working/train_masks")))


class CarvanaDataset():
    def __init__(self, image_dir, mask_dir, image_list, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Ensure mask is binary
        mask[mask>0] = 1.0

        # Apply Albumentations
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.float()

        return image, mask



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def train_fn(train_loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    loop = tqdm(train_loader, desc="Training", leave=True)

    for batch_idx, (imgs, targets) in enumerate(loop):
        imgs = imgs.to(device)
        targets = targets.float().unsqueeze(1).to(device) # required by BCE (B,) => (B,1)

        # forward
        with torch.autocast("cuda"):
            predictions = model(imgs)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        loop.set_postfix(loss=loss.item())
    return loss.item()


def val_fn(val_loader, model, device): 
    # evaluate by pixel accuracy and dice 
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    loop = tqdm(val_loader, desc="Validating", leave=True)

    with torch.no_grad():
        for imgs, targets in loop: 
            imgs = imgs.to(device) # (Batch_size, channels, h, w)
            targets = targets.to(device).unsqueeze(1) # (Batch_size, h, w) => (Batch_size, 1, h, w)
            preds = torch.sigmoid(model(imgs))
            # turn to binary mask
            preds = (preds > 0.5).float() # (>0.5)=>1, (<0.5)=>0 

            num_correct += (preds == targets).sum() # sum of all correctly predicted pixels
            num_pixels += torch.numel(preds) # sum of all pixels in the mask 
            
            intersection = (preds * targets).sum() # since both are binary mask, result=1 <=> both are 1
            union = (preds + targets).sum()
            dice_score += (2*intersection) / (union + 1e-8)
        dice_score = dice_score / len(val_loader)

    print(f"Correctly predicted {num_correct}/{num_pixels} => Acc: {(num_correct/num_pixels)*100:.2f}")
    print(f"Dice score: {dice_score}")

    return dice_score     


def freeze_backbone(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # Freeze all backbone params (ResNet) and set BN layers to eval mode
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    model.encoder.eval()
    print("Backbone frozen ~ Warm up stage")



def unfreeze_backbone(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # Unfreeze backbone for fine-tuning
    for name, param in model.encoder.named_parameters():
        param.requires_grad = True
    model.encoder.train()
    print("Backbone unfrozen ~ Fine-tuning stage")


def get_loaders(img_dir, mask_dir, train_img_list, val_img_list,
                batch_size, train_transforms, val_transforms,
                num_workers=4, pin_memory=True):
    
    train_dataset = CarvanaDataset(img_dir, mask_dir, train_img_list, train_transforms)
    val_dataset = CarvanaDataset(img_dir, mask_dir, val_img_list, val_transforms)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=True,
                             drop_last=True)
    val_loader = DataLoader(val_dataset, 
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False,
                             drop_last=True)
    return train_dataset, train_loader, val_dataset, val_loader
    


carvana_path = "/kaggle/working/"
trainval_images = os.path.join(carvana_path, "train_hq")
trainval_masks = os.path.join(carvana_path, "train_masks")
test_images = os.path.join(carvana_path, "test_hq")


seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_WORKERS = 4
IMAGE_HEIGHT = 1024 # 1280 originally
IMAGE_WIDTH = 1024  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "/kaggle/working/Semantic_Segmentation_UNet.pth"
IMG_DIR = trainval_images
MASK_DIR = trainval_masks


split_ratio = 0.9

all_images = [f for f in os.listdir(IMG_DIR)]
train_size = int(split_ratio * len(all_images))

random.seed(123)
random.shuffle(all_images)

train_image_list = all_images[:train_size]
val_image_list = all_images[train_size:]

print(f"({len(train_image_list)} train samples) | ({len(val_image_list)} validation samples)")


train_transforms = A.Compose([    
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # Stretch the longest side to 448px, then stretch the other side propotionally
            # Then add padding if the image still smaller after stretch

            # --- Augmentations ---
            A.HorizontalFlip(p=0.5), # randomly flip
            A.VerticalFlip(p=0.1),
            A.HueSaturationValue( hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4), #randomly adjust color - 40%
            A.RandomBrightnessContrast(p=0.4), # randomly adjust brightness - 40%
            A.Blur(blur_limit=3, p=0.2),
                
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            # randomly moves, zooms, rotates

            # --- Normalization ---
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )

val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


train_dataset, train_loader, val_dataset, val_loader = get_loaders(IMG_DIR, MASK_DIR, train_image_list, val_image_list,
                                    BATCH_SIZE, train_transforms, val_transforms,
                                    NUM_WORKERS, PIN_MEMORY)


model = UNetResnet34(in_channels=3, out_channels=1).to(DEVICE)
model = nn.DataParallel(model)
loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy + Sigmoid
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


scaler = torch.amp.GradScaler("cuda")
best_dice = 0

freeze_backbone(model)
for epoch in range(NUM_EPOCHS):
    print(f"[Epoch: {epoch+1}/{NUM_EPOCHS}]")
    train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    dice_score = val_fn(val_loader, model, device=DEVICE)
    if dice_score >= best_dice:
        best_dice = dice_score
        save_checkpoint(checkpoint)


optimizer = optim.AdamW(model.parameters(), lr=1e-5)

unfreeze_backbone(model)
for epoch in range(20):
    print(f"[Epoch: {epoch+1+NUM_EPOCHS}/{NUM_EPOCHS+20}]")
    train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    dice_score = val_fn(val_loader, model, device=DEVICE)
    if dice_score >= best_dice:
        best_dice = dice_score
        save_checkpoint(checkpoint)


import torch, gc

gc.collect()
torch.cuda.empty_cache()



if LOAD_MODEL: 
    checkpoint = torch.load(LOAD_MODEL_FILE, map_location="cpu")
    load_checkpoint(checkpoint, model, optimizer)


device = "cuda" if torch.cuda.is_available() else "cpu"


a, b = 8, 8

fig, axes = plt.subplots(a, b, figsize=(b*8, a*8))
random_indices = random.sample(range(len(val_dataset)), a * b)

idx = 0
for i in range(a):
    for j in range(b):

        img, _ = val_dataset[random_indices[idx]]
        idx += 1

        # add batch dimension
        img_input = img.unsqueeze(0).to(device)

        # model prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(img_input))[0, 0].cpu()  # remove batch + channel

        ax = axes[i][j]
        ax.imshow(pred, cmap="gray")
        ax.set_axis_off()

plt.tight_layout()
plt.show()



import torch
import matplotlib.pyplot as plt
import random
import numpy as np

# Resnet34 Normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)



def denormalize(img_tensor):
    """Undo normalization: (img * std + mean) and return numpy HWC"""
    img = img_tensor.clone().cpu()
    img = img * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


a, b = 10, 1  # number of samples shown = a

fig, axes = plt.subplots(a, 4, figsize=(4 * 6, a * 6))
random_indices = random.sample(range(len(val_dataset)), a)

for row, idx in enumerate(random_indices):

    img, gt_mask = val_dataset[idx]

    # ---- denormalize BEFORE converting to numpy ----
    img_np = denormalize(img)

    # ---- GT mask ----
    gt_mask_np = gt_mask.cpu().numpy()

    # ---- model prediction ----
    img_input = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_input))[0, 0].cpu().numpy()

    # ---- difference map ----
    diff = np.zeros((*gt_mask_np.shape, 3))

    fp = (pred > 0.5) & (gt_mask_np == 0)      # false positive (red)
    fn = (pred <= 0.5) & (gt_mask_np == 1)     # false negative (blue)
    ok = ~(fp | fn)                            # correct pixels (gray)

    diff[ok] = [0.7, 0.7, 0.7]
    diff[fp] = [1.0, 0.0, 0.0]
    diff[fn] = [0.0, 0.0, 1.0]

    # ---- plot ----
    axes[row][0].imshow(img_np)
    axes[row][0].set_title("Image (denormalized)")
    axes[row][0].set_axis_off()

    axes[row][1].imshow(gt_mask_np, cmap="gray")
    axes[row][1].set_title("Ground Truth Mask")
    axes[row][1].set_axis_off()

    axes[row][2].imshow(pred, cmap="gray")
    axes[row][2].set_title("Predicted Mask")
    axes[row][2].set_axis_off()

    axes[row][3].imshow(diff)
    axes[row][3].set_title("Error Map (FP=Red, FN=Blue)")
    axes[row][3].set_axis_off()

plt.tight_layout()
plt.show()



def create_submission_file(model, test_dir, output_csv, device):
    import cv2
    import os
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from tqdm import tqdm
    import torch

    ORIGINAL_W = 1918
    ORIGINAL_H = 1280

    model.eval()
    
    # Dataset for test images
    class TestDataset(Dataset):
        def __init__(self, folder):
            self.files = sorted(os.listdir(folder))
            self.folder = folder
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(),
            ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            fn = self.files[idx]
            img = cv2.imread(os.path.join(self.folder, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)["image"]
            return img, fn

    # Load test dataset
    test_ds = TestDataset(test_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    results = []

    with torch.no_grad():
        for imgs, fns in tqdm(test_loader):
            imgs = imgs.to(device)

            # forward pass
            preds = model(imgs)
            preds = torch.sigmoid(preds).cpu().numpy()  # [B,1,512,512]

            for mask, fn in zip(preds, fns):
                mask = mask[0]   # [512,512]

                # threshold
                mask = (mask > 0.5).astype(np.uint8)

                # ★ ★ ★ RESIZE BACK TO ORIGINAL SIZE ★ ★ ★
                mask = cv2.resize(mask, (ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_NEAREST)

                # encode RLE
                rle = mask_to_rle(mask)
                results.append([fn, rle])

    df = pd.DataFrame(results, columns=["img", "rle_mask"])
    df.to_csv(output_csv, index=False)



def mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

TEST_DIR = test_images
OUTPUT_CSV = "submission.csv"

create_submission_file(
    model=model,
    test_dir=TEST_DIR,
    output_csv=OUTPUT_CSV,
    device=device
)




