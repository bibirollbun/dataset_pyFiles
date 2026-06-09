import os
import random
from shutil import copy2
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets , transforms
from tqdm.notebook import tqdm


print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)
print("numpy version : ", np.__version__)

!python --version


df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
df.sample(10)


label_map = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}


data_dir = "/kaggle/input/cassava-leaf-disease-classification"


train_dir = os.path.join(data_dir , "train_images")
train_dir


output_path = os.path.join("/kaggle/working/" , "train")
for label in df['label'].unique():
    class_name = label_map[label]
    class_dir = os.path.join(output_path , class_name)
    os.makedirs(class_dir , exist_ok = True)


for _ , row in df.iterrows():
    filename = row['image_id']
    label = row['label']
    class_name = label_map[label]

    src_path = os.path.join(train_dir , filename)
    dst_path = os.path.join(output_path , class_name , filename)

    shutil.copy(src_path , dst_path)


classes = os.listdir(output_path)
classes


def sample_images(data_path , class_name):

    class_dir = os.path.join(data_path , class_name)
    
    if not os.path.exists(class_dir):
        return "Invalid directory"
        
    images_list = os.listdir(class_dir)

    random_imgs = random.sample(images_list , 4)

    #plot
    plt.figure(figsize = (20,20))
    
    for i in range(4):
        
        img_loc = os.path.join(class_dir , random_imgs[i])
        img = PIL.Image.open(img_loc)
        plt.subplot(1,4,i + 1)
        plt.imshow(img)
        plt.axis("off")


classes


sample_images(output_path , classes[0])


sample_images(output_path , classes[4])


sample_images(output_path , classes[2])


sample_images(output_path , classes[1])


sample_images(output_path , classes[3])


class ConvertToRGB(object):
    def __call__(self , img):
        if img.mode != "RGB":
            img =img.convert("RGB")
        return img


transform_basic = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]
)


batch_size = 32
dataset = datasets.ImageFolder(root = output_path , transform = transform_basic)
dataset_loader = DataLoader(dataset = dataset , batch_size = batch_size)
batch_shape = next(iter(dataset_loader))[0].shape
print("Getting batches of shape:", batch_shape)


def get_mean_std(loader):
    """Computes the mean and standard deviation of image data.

    Input: a `DataLoader` producing tensors of shape [batch_size, channels, pixels_x, pixels_y]
    Output: the mean of each channel as a tensor, the standard deviation of each channel as a tensor
            formatted as a tuple (means[channels], std[channels])"""

    channels_sum , channels_squared_sum , num_batches = 0 , 0 , 0
    
    for data , _ in tqdm(loader , desc = "Computing mean and std" , leave = True):
        channels_sum += torch.mean(data , dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2 , dim= [0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean , std


mean , std = get_mean_std(dataset_loader)
print(f"Mean = {mean}")
print(f"Standard deviation = {std}")


transform_norm = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean , std = std)
    ]
)


norm_dataset = datasets.ImageFolder(root = output_path , transform = transform_norm)
norm_loader = DataLoader(dataset = norm_dataset , batch_size = batch_size)
batch_shape = next(iter(norm_loader))[0].shape
print("Getting batches of shape", batch_shape)


norm_mean , norm_std = get_mean_std(norm_loader)
print(f"Mean = {norm_mean}")
print(f"Standard deviation = {norm_std}")


train_dataset , val_dataset = random_split(norm_dataset , [0.8,0.2])

length_train = len(train_dataset)
length_val = len(val_dataset)
length_dataset = len(norm_dataset)

percent_train = np.round(length_train * 100 / length_dataset , 2)
percent_val = np.round(length_val * 100 / length_dataset , 2)

print(f"Train data is {percent_train}% of full data")
print(f"Validation data is {percent_val}% of full data")


def class_count(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    try:
        class_to_index = dataset.class_to_idx
    except AttributeError:
        class_to_index = dataset.dataset.class_to_idx

    return pd.Series({cat: c[idx] for cat , idx in class_to_index.items()})


train_counts = class_count(train_dataset)
train_counts


train_counts.plot(kind="bar")


val_counts = class_count(val_dataset)
val_counts


val_counts.plot(kind= "bar")


def undersample_dataset(dataset_dir, output_dir, target_count=None):
    """
    Undersample the dataset to have a uniform distribution across classes.

    Parameters:
    - dataset_dir: Path to the directory containing the class folders.
    - output_dir: Path to the directory where the undersampled dataset will be stored.
    - target_count: Number of instances to keep in each class. If None, the class with the least instances will set the target.
    """
    # Mapping each class to its files
    classes_files = {}
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            files = os.listdir(class_dir)
            classes_files[class_name] = files

    # Determine the minimum class size if target_count is not set
    if target_count is None:
        target_count = min(len(files) for files in classes_files.values())

    # Creating the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perform undersampling
    for class_name, files in classes_files.items():
        print("Copying images for class", class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Randomly select target_count images
        selected_files = random.sample(files, min(len(files), target_count))

        # Copy selected files to the output directory
        for file_name in tqdm(selected_files):
            src_path = os.path.join(dataset_dir, class_name, file_name)
            dst_path = os.path.join(class_output_dir, file_name)
            copy2(src_path, dst_path)

    print(f"Undersampling completed. Each class has up to {target_count} instances.")


output_sampled_dir = os.path.join("/kaggle/working/" , "data_undersampled", "train")
print("Output directory:", output_sampled_dir)


undersample_dataset(output_path, output_sampled_dir)


undersampled_dataset = datasets.ImageFolder(root = output_sampled_dir , transform= transform_norm)


undersampled_dataset.classes


undersample_counts = class_count(undersampled_dataset)


fig , ax = plt.subplots(figsize = (10,7))
print(undersample_counts)
undersample_counts.plot(kind = "bar" , ax=ax);

