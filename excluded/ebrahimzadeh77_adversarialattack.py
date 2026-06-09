name = "AlirezaEbrahimzadeh"
student_id = "402204244"


### Note : THis code should be run on kaggle (last about 1H)
### Note : Before run , u have to add ILSVRC dataset to ur input dataset , (https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)
### for add dataset Type "imagenet object localization challenge"


## Standard libraries
import os
import json
import math
import time
import numpy as np
import scipy.linalg

## Imports for plotting
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    !pip install --quiet pytorch-lightning>=1.4
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "/kaggle/working/"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/kaggle/working/"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)


import os
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET

VAL_IMAGES_DIR = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
ANNOTATIONS_DIR = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val"
OUTPUT_DIR_VAL = "/kaggle/working/val"


class_to_images = defaultdict(list)

for xml_file in sorted(os.listdir(ANNOTATIONS_DIR)):
    if not xml_file.endswith('.xml'):
        continue
    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image filename
    filename = root.find("filename").text

    # Extract class label (first <object><name>)
    object_elem = root.find("object")
    if object_elem is not None:
        class_name = object_elem.find("name").text
        class_to_images[class_name].append(filename)

# Create output directory
os.makedirs(OUTPUT_DIR_VAL, exist_ok=True)

for class_name, image_list in class_to_images.items():
    output_class_dir = os.path.join(OUTPUT_DIR_VAL, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in sorted(image_list)[:]:
        src_img_path = os.path.join(VAL_IMAGES_DIR, img_name+".JPEG")
        dst_img_path = os.path.join(output_class_dir, img_name+".JPEG")
        shutil.copy(src_img_path, dst_img_path)




from torchvision import models


pretrained_model = models.resnet34(pretrained=True)

# no gradiant
for param in pretrained_model.parameters():
    param.requires_grad = False


for param in pretrained_model.fc.parameters():
    param.requires_grad = True

# Move to GPu
pretrained_model = pretrained_model.to(device)

print(pretrained_model)



#resize tensor to uniform size

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#create dataset and dataloader
dataset = torchvision.datasets.ImageFolder(OUTPUT_DIR_VAL,transform=transform)
dataloader = data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

#calculate mean and std
mean = 0.0
std = 0.0
nb_samples = 0

for data, _ in dataloader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # flatten H and W
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples



print("Mean:", mean)
print("Std:", std)


import torch.utils.data as data


file_path = '/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt'
with open(file_path, 'r') as file:
    file_content = file.read()

#create dic of labels dic

label_images = {}
for idx, line in enumerate(file_content.strip().split("\n")):
    parts = line.split()
    wnid = parts[0]  # WordNet ID
    labels = " ".join(parts[1:]).split(", ")  # Split labels
    label_images[idx] = [wnid] + labels

for key in label_images:
    label_images[key] = label_images[key][1:]



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
os.makedirs("/kaggle/working/val_subset", exist_ok=True)


N_IMAGES = 5


for class_name in sorted(os.listdir("/kaggle/working/val")):
    class_input_path = os.path.join("/kaggle/working/val", class_name)
    class_output_path = os.path.join("/kaggle/working/val_subset", class_name)

    os.makedirs(class_output_path, exist_ok=True)

    image_files = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in sorted(image_files)[:N_IMAGES]:
        src = os.path.join(class_input_path, img_file)
        dst = os.path.join(class_output_path, img_file)
        shutil.copy(src, dst)
# create sub dataloader that have 5 pics of each 1000 classes
sub_dataset = torchvision.datasets.ImageFolder("/kaggle/working/val_subset", transform=transform)
sub_dataloader = data.DataLoader(sub_dataset, batch_size=128, shuffle=True, num_workers=4)


#evaluate model
def eval_model(dataset_loader, img_func=None):
    model.eval()
    correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataset_loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            _, preds = output.topk(5, dim=1)

            # Top-1 accuracy
            correct += (preds[:, 0] == labels).sum().item()
            # Top-5 accuracy
            top5_correct += sum([labels[i] in preds[i] for i in range(labels.size(0))])
            total += labels.size(0)

    acc = correct / total
    top5 = top5_correct / total

    print(f"Top-1 error: {(1-acc)*100:.2f}%")
    print(f"Top-5 error: {(1-top5)*100:.2f}%")
    return acc, top5



model=pretrained_model
_ = eval_model(sub_dataloader)


def show_prediction(img, label, pred, K=5, adv_img=None, noise=None):

    img_np = img.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * np.array(std) + np.array(mean))  # Unnormalize
    img_np = np.clip(img_np, 0, 1)

    # Top K predictions
    top_probs, top_classes = torch.topk(F.softmax(pred, dim=0), K)
    top_probs = top_probs.detach().cpu().numpy()
    top_classes = top_classes.detach().cpu().numpy()

    # Set up figure
    fig, axs = plt.subplots(1, 3 if adv_img is not None else 2, figsize=(14, 4))

    # Original image
    axs[0].imshow(img_np)
    axs[0].axis('off')
    axs[0].set_title(f"True label: {label}")

    # Top-K predictions
    class_labels = [str(cls) for cls in top_classes]
    axs[1].barh(range(K), top_probs)
    axs[1].set_yticks(range(K))
    axs[1].set_yticklabels(class_labels)
    axs[1].invert_yaxis()
    axs[1].set_title("Top-K Predictions")

    # Optional: Adversarial image and noise
    if adv_img is not None and noise is not None:
        adv_np = adv_img.cpu().permute(1, 2, 0).numpy()
        adv_np = (adv_np * np.array(std) + np.array(mean))
        adv_np = np.clip(adv_np, 0, 1)

        noise_np = noise.cpu().permute(1, 2, 0).numpy()
        noise_np = noise_np / (2 * np.abs(noise_np).max()) + 0.5  # Normalize to [0,1] for visualization

        axs[2].imshow(adv_np)
        axs[2].axis('off')
        axs[2].set_title("Adversarial Image")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(noise_np)
        ax.axis('off')
        ax.set_title("Noise")

    plt.tight_layout()
    plt.show()



exmp_batch, label_batch = next(iter(sub_dataloader))
with torch.no_grad():
    preds = pretrained_model(exmp_batch.to(device))
for i in range(1,17,4):
    show_prediction(exmp_batch[i], label_batch[i], preds[i])
    


def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):


    model.eval()

    # copy and prepare inputs
    imgs = imgs.clone().detach().to(device)
    labels = labels.to(device)
    imgs.requires_grad = True  # We want gradients w.r.t. the input

    # forward pass
    outputs = model(imgs)
    log_probs = F.log_softmax(outputs, dim=1)

    # compute the loss
    loss = F.nll_loss(log_probs, labels)

    # backward pass to compute gradients
    model.zero_grad()
    loss.backward()

    # compute perturbation using sign of gradients
    noise_grad = imgs.grad.data
    perturbation = epsilon * noise_grad.sign()

    # create adversarial examples 
    adv_imgs = imgs + perturbation

    return adv_imgs.detach(), noise_grad.detach()



adv_imgs, noise_grad = fast_gradient_sign_method(pretrained_model, exmp_batch, label_batch, epsilon=0.02)
with torch.no_grad():
    adv_preds = pretrained_model(adv_imgs.to(device))

for i in range(1,17,4):
    show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=adv_imgs[i], noise=noise_grad[i])


_ = eval_model(sub_dataloader, img_func=lambda x, y: fast_gradient_sign_method(pretrained_model, x, y, epsilon=0.02)[0])


def place_patch(img, patch):
    
    C, H, W = img.shape
    _, h, w = patch.shape

    # patch size smaller than pic
    assert h <= H and w <= W

    # random position
    top = np.random.randint(0, H - h + 1)
    left = np.random.randint(0, W - w + 1)

    # copy original image
    img_patched = img.clone()

    # cverlay patch on image at the selected location
    img_patched[:, top:top + h, left:left + w] = patch

    return img_patched



TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(mean)[:,None,None], torch.FloatTensor(std)[:,None,None]
def patch_forward(patch):
    
    # sigmoid to squash into [0,1]
    patch_img = torch.sigmoid(patch)

    # normalize using imageNet statistics
    patch_img = (patch_img - TENSOR_MEANS.to(patch.device)) / TENSOR_STD.to(patch.device)

    return patch_img


def eval_patch(model, patch, val_loader, target_class):
    """
    Evaluate how often the model is fooled into predicting the target class due to the patch.
    """
    model = model.to(device)
    model.eval()
    total, fooled_top1, fooled_top5 = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # skip images that are already of the target class
            mask = labels != target_class
            if mask.sum() == 0:
                continue
            imgs, labels = imgs[mask], labels[mask]

            batch_size = imgs.size(0)
            preds_1 = torch.zeros(batch_size, device=device)
            preds_5 = torch.zeros(batch_size, device=device)

            for _ in range(4):
                patch_img = patch_forward(patch).detach().to(device)
                patched_imgs = torch.stack([
                    place_patch(img, patch_img) for img in imgs
                ]).to(device)

                outputs = model(patched_imgs)
                top1_preds = outputs.argmax(dim=1)
                top5_preds = torch.topk(outputs, 5, dim=1).indices

                preds_1 += (top1_preds == target_class).float()
                preds_5 += (top5_preds == target_class).any(dim=1).float()

            fooled_top1 += (preds_1 >= 1).sum().item()
            fooled_top5 += (preds_5 >= 1).sum().item()
            total += batch_size

    acc = fooled_top1 / total
    top5 = fooled_top5 / total
    return acc, top5



def patch_attack(model, target_class, patch_size=64, num_epochs=5):
    
    model.eval()

    # use 90% of data for training patch, 10% for evaluation
    train_len = int(len(sub_dataloader.dataset) * 0.9)
    val_len = len(sub_dataloader.dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(sub_dataloader.dataset, [train_len, val_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

    # initialize patch parameter
    patch = nn.Parameter(torch.randn(3, patch_size, patch_size))

    # optimizer (AdamW)
    optimizer = torch.optim.AdamW([patch], lr=0.005, weight_decay=0.001,)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # skip target class images (not fooling them)
            mask = labels != target_class
            if mask.sum() == 0:
                continue
            imgs = imgs[mask]

            # forward: place patch
            patch_img = patch_forward(patch)
            patched_imgs = torch.stack([place_patch(img, patch_img) for img in imgs])

            # predict and calculate loss
            outputs = model(patched_imgs)
            target_labels = torch.full((patched_imgs.size(0),), target_class, dtype=torch.long, device=device)
            loss = loss_fn(outputs, target_labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

    # final validation
    acc, top5 = eval_patch(model, patch, val_loader, target_class)
    return patch.data, {"acc": acc, "top5": top5}



# load evaluation results of the pretrained patches
json_results_file = os.path.join(CHECKPOINT_PATH, "patch_results.json")
json_results = {}
if os.path.isfile(json_results_file):
    with open(json_results_file, "r") as f:
        json_results = json.load(f)

# if you train new patches, you can save the results via calling this function
def save_results(patch_dict):
    result_dict = {cname: {psize: [t.item() if isinstance(t, torch.Tensor) else t
                                   for t in patch_dict[cname][psize]["results"]]
                           for psize in patch_dict[cname]}
                   for cname in patch_dict}
    with open(os.path.join(CHECKPOINT_PATH, "patch_results.json"), "w") as f:
        json.dump(result_dict, f, indent=4)


def search_in_dict(value, label_images):
    # iterate over dictionary to find the key for a given value
    for key, values in label_images.items():
        if value in values:
            return key
    return None  # If value not found, return None


def get_patches(class_names, patch_sizes):
    result_dict = dict()

    for cname in class_names:
        result_dict[cname] = {}
        target_class = search_in_dict(cname, label_images)  # convert name to ImageNet class index

        for psize in patch_sizes:
            print(f"Processing class '{cname}' with patch size {psize}...")

            patch_file = os.path.join(CHECKPOINT_PATH, f"patch_{cname}_{psize}.pt")
            patch_results = json_results.get(cname, {}).get(str(psize), None)

            if os.path.isfile(patch_file):
                patch_tensor = torch.load(patch_file, map_location=device)
            else:
                patch_tensor, stats = patch_attack(model, target_class, patch_size=psize)
                torch.save(patch_tensor, patch_file)
                print(f"Trained and saved patch for class '{cname}', size {psize}")
                patch_results = [stats["acc"], stats["top5"]]

            # If patch file existed but results not yet available, evaluate
            if patch_results is None:
                acc, top5 = eval_patch(model, patch_tensor, sub_dataloader, target_class)
                patch_results = [acc, top5]
                print(f"Manually evaluated patch for class '{cname}', size {psize}")

            # Store in result dict
            result_dict[cname][str(psize)] = {
                "patch": patch_tensor,
                "results": patch_results
            }

    return result_dict



class_names = ['toaster', 'goldfish', 'school bus', 'lipstick', 'pineapple']
patch_sizes = [32, 48, 64]
patch_dict = get_patches(class_names, patch_sizes)
save_results(patch_dict) # Uncomment if you add new class names and want to save the new results


import matplotlib.pyplot as plt

def show_patches(patch_dict):
    """
    Visualizes trained adversarial patches from the patch dictionary.

    Args:
        patch_dict: Dictionary containing patches, keyed by class name and patch size.
    """
    num_classes = len(patch_dict)
    num_sizes = len(next(iter(patch_dict.values())))
    
    fig, axes = plt.subplots(num_classes, num_sizes, figsize=(3 * num_sizes, 3 * num_classes))

    for row_idx, (class_name, size_dict) in enumerate(patch_dict.items()):
        for col_idx, (patch_size, patch_info) in enumerate(size_dict.items()):
            patch = patch_forward(patch_info["patch"]).detach().cpu()
            patch_img = patch.permute(1, 2, 0).numpy()
            patch_img = (patch_img - patch_img.min()) / (patch_img.max() - patch_img.min())  # Normalize for display

            ax = axes[row_idx, col_idx] if num_classes > 1 else axes[col_idx]
            ax.imshow(patch_img)
            ax.axis('off')
            acc, top5 = patch_info["results"]
            ax.set_title(f"{class_name}\n{patch_size}px\nTop1: {acc:.2%}, Top5: {top5:.2%}")

    plt.tight_layout()
    plt.show()

show_patches(patch_dict)


%%html
<!-- Some HTML code to increase font size in the following table -->
<style>
th {font-size: 120%;}
td {font-size: 120%;}
</style>


import tabulate
from IPython.display import display, HTML

def show_table(top_1=True):
    i = 0 if top_1 else 1
    table = [[name] + [f"{(100.0 * patch_dict[name][str(psize)]['results'][i]):4.2f}%" for psize in patch_sizes]
             for name in class_names]
    display(HTML(tabulate.tabulate(table, tablefmt='html', headers=["Class name"] + [f"Patch size {psize}x{psize}" for psize in patch_sizes])))


show_table(top_1=True)


show_table(top_1=False)


import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def perform_patch_attack(patch, model, val_loader, target_class, device='cuda', num_images=3):
    model = model.to(device).eval()
    patch = patch.to(device)
    patch_img = patch_forward(patch).detach()

    images_shown = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            for idx in range(imgs.size(0)):
                if labels[idx].item() == target_class:
                    continue  # skip true target class

                img = imgs[idx]
                label = labels[idx].item()

                # Apply patch
                patched = place_patch(img, patch_img)

                # Get predictions
                orig_out = model(img.unsqueeze(0))
                patch_out = model(patched.unsqueeze(0))

                orig_pred = orig_out.argmax(dim=1).item()
                patch_pred = patch_out.argmax(dim=1).item()

                # Convert for plotting
                unnorm = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                                              std=[1/s for s in std])
                img_disp = TF.to_pil_image(unnorm(img.cpu()))
                patch_disp = TF.to_pil_image(unnorm(patched.cpu()))

                # Plotting
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img_disp)
                axs[0].set_title(f"Original: {label_images.get(orig_pred)[0]}")
                axs[0].axis('off')

                axs[1].imshow(patch_disp)
                axs[1].set_title(f"Patched: {label_images.get(patch_pred)[0]}")
                axs[1].axis('off')

                plt.tight_layout()
                plt.show()

                images_shown += 1
                if images_shown >= num_images:
                    return



perform_patch_attack(
    patch=patch_dict['goldfish'][str(32)]['patch'],
    model=model,
    val_loader=sub_dataloader,
    target_class=search_in_dict('goldfish', label_images)
)



perform_patch_attack(
    patch=patch_dict['school bus'][str(64)]['patch'],
    model=model,
    val_loader=sub_dataloader,
    target_class=search_in_dict('school bus', label_images)
)


from torchvision import models

# load DenseNet121 pretrained on ImageNet
transfer_model = models.densenet121(pretrained=True)

# don’t compute or store gradients
for param in transfer_model.parameters():
    param.requires_grad = False

transfer_model = transfer_model.to(device)
transfer_model.eval()

print(transfer_model)


class_name = 'pineapple'
patch_size = 64
print(f"Testing patch \"{class_name}\" of size {patch_size}x{patch_size}")

results = eval_patch(transfer_model,
                     patch_dict[class_name][str(patch_size)]["patch"],
                     val_loader=sub_dataloader,
                     target_class=search_in_dict(class_name, label_images))

print(f"Top-1 fool accuracy: {(results[0] * 100.0):4.2f}%")
print(f"Top-5 fool accuracy: {(results[1] * 100.0):4.2f}%")

