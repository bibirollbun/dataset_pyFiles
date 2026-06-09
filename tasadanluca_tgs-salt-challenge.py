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


if not os.path.exists('train_data'):
    !unzip -q /kaggle/input/tgs-salt-identification-challenge/train.zip -d train_data


if not os.path.exists('test_data'):
    !unzip -q /kaggle/input/tgs-salt-identification-challenge/test.zip -d test_data




image_dir = '/kaggle/working/train_data/images'
mask_dir = '/kaggle/working/train_data/masks'

print(len(os.listdir(image_dir)))
print(len(os.listdir(mask_dir)))


import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader


RESIZE_SIZE = 202
FINAL_SIZE = 256
train_transform = A.Compose([
    # First, resize the image to 202x202
    A.Resize(height=RESIZE_SIZE, width=RESIZE_SIZE, p=1.0),
    # Pad the 202x202 to 256x256 in order for the pooling and upscaling from the
    # Unet to work properly 
    A.PadIfNeeded(min_height=FINAL_SIZE, min_width=FINAL_SIZE, p=1.0),
    # Apply one of 8 rand symmetries
    A.SquareSymmetry(p=1.0),
    # Image-only transforms
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(std_range=(0.1,0.2),p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std = (0.229,0.224,0.225)),
    A.ToTensorV2(),
])

val_transform = A.Compose([
    # Must apply the same to the validation transformation
    A.Resize(height=RESIZE_SIZE, width=RESIZE_SIZE, p=1.0),
    A.PadIfNeeded(min_height=FINAL_SIZE, min_width=FINAL_SIZE, p=1.0),
    A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
    A.ToTensorV2(),
])


data_filenames = os.listdir(image_dir)
#train_filenames, val_filenames = train_test_split(data_filenames,test_size = 0.2, random_state = 42)



image_mask = Image.open(os.path.join(mask_dir,data_filenames[5])).convert('L')
image = Image.open(os.path.join(image_dir,data_filenames[5])).convert('L')

image_np = np.array(image)
mask_np = np.array(image_mask)
print(image_np.shape, mask_np.shape)

plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.show()
plt.imshow(image_mask, cmap = 'gray')
plt.axis('off')
plt.show()




augmented = train_transform(image = image_np, mask = mask_np)
augmented_image = augmented['image']
augmented_mask = augmented['mask']

print(f"Augmented image shape: {augmented_image.shape}, dtype: {augmented_image.dtype}")
print(f"Augmented mask shape: {augmented_mask.shape}, dtype: {augmented_mask.dtype}")
# I have to unsqueeze the mask so that it has the same shape as the image


class SALTimagesdataset(Dataset):
    def __init__(self,image_dir, mask_dir, files, transform = None):
        self.image_paths = image_dir
        self.mask_paths = mask_dir
        self.files = files
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_name = self.files[idx]
        image_path = os.path.join(self.image_paths,image_name)
        # Read the images
        image = cv2.imread(image_path)
        # Even though the images are grey scale, they must be converted to 
        # rgb because I will be using a pretrained model that was trained on rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # The mask should remain a grayscale as there is no need for rgb

        if self.mask_paths is not None:
                mask_path = os.path.join(self.mask_paths,image_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask / 255.0 # Normalize + convert to float32
        else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        
        if self.mask_paths is not None:
            # For training and validation, return image and mask
            return image, mask
        else:
            # For testing, return the image and its ID for the submission file
            return image, image_name



def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)): # Green overlay
    # Convert mask to 3 channels if needed, ensure boolean type
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    # Create a color overlay where mask is > 0
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)

    # Blend image and overlay
    overlayed_image = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image
    
def visualize_segmentation(dataset, idx = 0, samples = 3):
    # Make a copy of the transform list t omodify for visualization
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform

    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))

    # --- Get the original image and mask --- #
    original_transform = dataset.transform
    dataset.transform = None # Temporarily disable for raw data access
    image, mask = dataset[idx]
    dataset.transform = original_transform # Restore

    # Display original
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    #ax[0, 1].imshow(mask, cmap='gray') # Show mask directly
    #ax[0, 1].set_title("Original Mask")
    ax[0, 1].axis("off")
    ax[0, 1].imshow(overlay_mask(image, mask)) # Or show overlay
    ax[0, 1].set_title("Original Overlay")

    # --- Apply and display augmented versions --- #
    for i in range(samples):
        # Apply the visualization transform
        if vis_transform:
            augmented = vis_transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
        else:
            aug_image, aug_mask = image, mask # Should not happen normally

        # Display augmented image and mask
        ax[i + 1, 0].imshow(aug_image)
        ax[i + 1, 0].set_title(f"Augmented Image {i+1}")
        ax[i + 1, 0].axis("off")

        #ax[i + 1, 1].imshow(aug_mask, cmap='gray') # Show mask directly
        #ax[i + 1, 1].set_title(f"Augmented Mask {i+1}")
        ax[i + 1, 1].axis("off")
        ax[i+1, 1].imshow(overlay_mask(aug_image, aug_mask)) # Or show overlay
        ax[i+1, 1].set_title(f"Augmented Overlay {i+1}")


    plt.tight_layout()
    plt.show()


# train_dataset = SALTimagesdataset(image_dir, mask_dir, train_filenames, transform = train_transform)
# val_dataset = SALTimagesdataset(image_dir, mask_dir, val_filenames, transform = val_transform)
# print(len(train_dataset))

# The whole dataset with train transformations
train_dataset_full = SALTimagesdataset(image_dir, mask_dir, data_filenames, transform = train_transform) 
# The whole dataset with val transformations
val_dataset_full = SALTimagesdataset(image_dir, mask_dir, data_filenames, transform = val_transform) 
print(len(val_dataset_full))
print(len(data_filenames))


visualize_segmentation(train_dataset_full)


# Use 4 workers because the CPU from Kaggle Notebook has 4 cores
# train_loader =  DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 4)
# val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False, num_workers = 4)


try:
    # Try to import the library
    import segmentation_models_pytorch as smp
    print("segmentation_models_pytorch is already installed.")
except ImportError:
    # If it fails, run the installation
    print("Installing segmentation_models_pytorch...")
    !pip install -q segmentation_models_pytorch
    print("Installation complete.")
    import segmentation_models_pytorch as smp


def get_base_efficientnetb4_model():
    model = smp.Unet(
        encoder_name = "efficientnet-b4",
        encoder_weights = "imagenet",
        in_channels = 3,
        classes = 1,
    )
    return model



def calculate_iou(pred_mask, true_mask, salt_pred_threshold = 0.5):
    pred_mask_binary = (pred_mask > salt_pred_threshold).float()

    # Handle the cases where one of the masks doesn't have any salt pixels
    is_true_empty = true_mask.sum() == 0
    is_pred_empty = pred_mask_binary.sum() == 0

    if is_true_empty and is_pred_empty:
        return 1.0  # Correctly predicted empty mask
    if is_true_empty and not is_pred_empty:
        return 0.0  # False positive: predicted salt where there is none
    if not is_true_empty and is_pred_empty:
        return 0.0  # False negative: missed salt that was present

    intersection = (pred_mask_binary * true_mask).sum()
    union = pred_mask_binary.sum() + true_mask.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    iou_thresholds = np.linspace(0.5,0.95,10) # Takes 10 values evenly spaced between 0,5 and 0,95
    
    # How many thresholds does the pred_mask pass
    hits = (iou.item() > iou_thresholds).sum()

    mean_iou = hits/len(iou_thresholds)

    return mean_iou
    


import copy
from torch.cuda.amp import GradScaler
from torch.amp import autocast 
def train_model(model, device, train_loader, val_loader, 
            loss_function, optimizer, scheduler, 
                num_epochs, one_cycle = False, patience = 15):
    history = {
        'train_loss':[],
        'val_loss':[],
        'train_iou':[],
        'val_iou':[],
        'lr':[]
    }

    scaler = torch.amp.GradScaler('cuda')
    
    epochs_no_improve = 0    
    best_model = copy.deepcopy(model.state_dict())
    best_iou = 0.0
    for epoch in range(num_epochs):
        print("\n")
        print(f"Current Epoch:{epoch+1}/{num_epochs}")

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            running_loss = 0.0
            running_iou  = 0.0
    
            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)
                # Set gradients to 0
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(device_type='cuda'):
                        out = model(images)
                        predictions = torch.sigmoid(out)
                        loss = loss_function(out, masks)
                        
                    # If in training phase update gradients
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        if one_cycle:
                            scheduler.step()
                            
                    running_loss += loss.item() * images.size(0)

                    for pred,mask in zip(predictions,masks):
                        running_iou += calculate_iou(pred, mask)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_iou / len(dataloader.dataset)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_iou'].append(epoch_acc)
                history['lr'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_loss'].append(epoch_loss)
                history['val_iou'].append(epoch_acc)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f} Lr:{optimizer.param_groups[0]['lr']:.2e}")

            if phase == 'val':
                if epoch_acc > best_iou:
                    print(f"Validation IoU improved from {best_iou:.4f} to {epoch_acc:.4f}. Saving model.")

                    best_iou = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"Validation IoU didn't improve, Patience: {epochs_no_improve}/{patience}")
        if scheduler and not one_cycle:
            scheduler.step()
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break        
    print(f"Best validation accuracy:{best_iou}")
    model.load_state_dict(best_model)
    return model, history


                    


def plot_history(history):
    """
    Plots the training and validation loss, IoU, and learning rate.
    """
    # Create a 1x3 grid of subplots
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot loss
    ax[0].plot(history['train_loss'], label='Train Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot iou (accuracy)
    ax[1].plot(history['train_iou'], label='Train IoU')
    ax[1].plot(history['val_iou'], label='Validation IoU')
    ax[1].set_title('IoU vs. Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('IoU')
    ax[1].legend()
    ax[1].grid(True)
    
    # Plot lr
    ax[2].plot(history['lr'], label='Learning Rate')
    ax[2].set_title('Learning Rate vs. Epochs')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Learning Rate')
    ax[2].legend()
    ax[2].grid(True)
    plt.tight_layout()
    plt.show()



import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message='"is" with a literal. Did you mean "=="?')

file_name = "lovasz_losses.py"

if not os.path.exists(file_name):
    print("Downloading lovasz_losses.py...")
    # Use wget to download the  Python file from GitHub
    !wget -q https://raw.githubusercontent.com/bermanmaxim/LovaszSoftmax/master/pytorch/lovasz_losses.py
    print("Download complete.")
else:
    print(f"{file_name} already exists.")

from lovasz_losses import lovasz_hinge


dice_loss = smp.losses.DiceLoss(mode = 'binary')
bce_loss = nn.BCEWithLogitsLoss()

def bce_with_dice_loss(inputs, targets, weight_bce =0.5, weight_dice = 0.5):
    loss_bce = bce_loss(inputs,targets)
    loss_dice = dice_loss(inputs,targets)
    return (weight_bce * loss_bce) + (weight_dice * loss_dice)


from tqdm import tqdm
import cv2
import numpy as np
# I have to know the images that have no salt so that my 5 fold model will not be imbalanced
# as the images with no salt can negatively impact the performance of my model
y_stratify = []


print("Generating stratification labels")
for mask_path in tqdm(data_filenames):
    full_path = os.path.join(mask_dir, mask_path)
    mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)    
    if np.sum(mask) > 0:
        y_stratify.append(1)
    else:
        y_stratify.append(0)
y_stratify = np.array(y_stratify)

print(f"Stratification labels created. Found {np.sum(y_stratify)} salt masks.")


# First I will train only on the last layer of resnet34, then I will train on the decoder
# I will use a 5-fold model
from torch import optim
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import Subset

num_splits = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = 42)

histories = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(data_filenames, y_stratify)):

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, val_idx)
    #print(len(train_subset), len(val_subset))
    train_loader = DataLoader(train_subset, batch_size = 32, shuffle = True,
                              num_workers = 4, pin_memory = True)
    val_loader = DataLoader(val_subset, batch_size =32, shuffle = False,
                            num_workers = 4, pin_memory = True)

    #print(len(train_loader),len(val_loader))
    
    model = get_base_efficientnetb4_model()

    model = model.to(device)
    
    print("Decoder only training: \n")
    
    # Freeze parameters 
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    learning_rate_decoder = 1e-4
    
    params_to_train = [param for param in model.parameters() if param.requires_grad == True]
    
    optimizer_decoder = optim.Adam(params_to_train,lr = learning_rate_decoder)
    
    scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_decoder,
    T_max = 50,
    eta_min = 1e-5,
)
    
    
    loss_function_decoder = bce_with_dice_loss
    
    decoder_best_model, decoder_history = train_model(model, device, train_loader, val_loader, loss_function_decoder,
                                              optimizer_decoder,
                                              scheduler_decoder,
                                              num_epochs = 50)
    plot_history(decoder_history)
    
    print("Full network training: \n")
    
    model.load_state_dict(decoder_best_model.state_dict())

    # Unfreeze the parameters from the encoder
    for param in model.parameters():
        param.requires_grad = True
    
    learning_rate_full = 5e-5
    
    
    optimizer_full = optim.Adam(model.parameters(), lr = learning_rate_full)
    
    scheduler_full = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_full,
        max_lr=learning_rate_full,
        steps_per_epoch=len(train_loader), # Number of batches in the training loader
        epochs=50,
        pct_start=0.3, # Use 30% of steps for the warm-up phase
    )
    
    loss_function_full = lovasz_hinge
    
    
    full_best_model, full_history = train_model(model, device, train_loader, val_loader, loss_function_full,
                                           optimizer_full,
                                           scheduler_full,
                                           num_epochs = 50,
                                          one_cycle = True)
    plot_history(full_history)

    print(f"Saving results for fold {fold+1}")
    best_fold_score = max(full_history['val_iou'])
    fold_scores.append(best_fold_score)

    torch.save(full_best_model.state_dict(),f"best_model_fold{fold+1}.pth")

    print(f"Fold {fold + 1} Best Score: {best_fold_score:.4f}\n")


print("SUMMARY: ")
mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
print(f"Average Score over {num_splits} Folds: {mean_score:.4f} ± {std_score:.4f}")



models = []
num_splits = 5

for fold in range(num_splits):
    # Create a fresh model instance
    model = get_base_efficientnetb4_model()

    # Load the saved weights for the current fold
    model_path = f"best_model_fold{fold+1}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} models for ensembling.")


test_image_dir = '/kaggle/working/test_data/images'
test_filenames = os.listdir(test_image_dir)
test_dataset = SALTimagesdataset(test_image_dir, None, test_filenames, transform = val_transform)


#torch.save(best_model.state_dict(), 'best_model_weights2.pth')



test_loader = DataLoader(test_dataset, batch_size =32, shuffle = False, num_workers = 4)
#best_model.eval()


from tqdm import tqdm
predictions = []
test_ids = []
with torch.no_grad():
    for images,ids in tqdm(test_loader, desc = "Predicting"):
        images = images.to(device, dtype = torch.float32)
        
        fold_probs = []
        for model in models:
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            fold_probs.append(probs)

        # Average the predictions
        avg_probs = torch.stack(fold_probs).mean(dim=0)

        #  Threshold the mean probabilities
        masks = (avg_probs.squeeze() > 0.5).cpu().numpy().astype(np.uint8)

        predictions.extend(list(masks))
        test_ids.extend(ids)


# Encodes a binary mask into an encoding string, the mask should be 2D numpy array
def rle_encode(mask):
    if np.sum(mask) == 0:
        return ""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0],pixels,[0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2]-=runs[::2]
    return ' '.join(str(x) for x in runs)



test_ids_clean = test_ids_clean = [os.path.splitext(name)[0] for name in test_ids]
# Define the original and padded sizes
ORIGINAL_SIZE = 101
RESIZED_SIZE = 202
PADDED_SIZE = 256 
# Calculate the slice boundaries
start = (PADDED_SIZE - RESIZED_SIZE) // 2
end = start + RESIZED_SIZE

rle_predictions = []
for p in predictions:
    # Crop the padded area to get the 202x202 mask
    cropped_mask = p[start:end, start:end]

    # Resize the cropped mask back to the original 101x101 size
    final_mask = cv2.resize(cropped_mask, (ORIGINAL_SIZE, ORIGINAL_SIZE), interpolation=cv2.INTER_LINEAR)

    # Encode the final mask
    rle_predictions.append(rle_encode(final_mask))


# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids_clean, # Name of the file
    'rle_mask': rle_predictions # The predicted salt patches
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully: submission.csv")




