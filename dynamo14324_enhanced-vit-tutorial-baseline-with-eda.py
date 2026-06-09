import os, sys, warnings
warnings.filterwarnings('ignore')
# os.environ['KAGGLE_CONFIG_DIR']='/root/.kaggle' # Usually not needed inside Kaggle env
# sys.path.append('/kaggle/working') # Usually not needed inside Kaggle env
print('Initial setup code executed.')


!cp -r /kaggle/input/vittutorialillustrations/* ./ 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import torch
import torch.nn as nn
import torchvision.transforms as transforms


import timm

import gc
import os
import time
import random
from datetime import datetime

from PIL import Image
from tqdm.notebook import tqdm
from sklearn import model_selection, metrics


def seed_everything(seed):
    # Seeds basic parameters for reproducibility across runs
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1001)


# general global variables
# Standard Kaggle input path for the competition data
DATA_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_PATH = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_PATH = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
MODEL_PATH = (
    "/kaggle/input/vit-base-models-pretrained-pytorch/jx_vit_base_p16_224-80ecf9dd.pth"
)

# model specific global variables
IMG_SIZE = 224
BATCH_SIZE = 16
LR = 2e-05
GAMMA = 0.7
N_EPOCHS = 10


df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df.head()


try:
    # Visualize Class Distribution
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('Distribution of Cassava Leaf Disease Classes')
    # Map numeric labels to actual names if available (using the competition's label map)
    # label_map = {0: 'Cassava Bacterial Blight (CBB)', 1: 'Cassava Brown Streak Disease (CBSD)', ...}
    # plt.xticks(ticks=range(len(label_map)), labels=label_map.values(), rotation=45, ha='right') # Example if label map exists
    plt.xlabel('Disease Label')
    plt.ylabel('Number of Images')
    plt.show()
    
    # Print value counts
    print("Class Counts:\n", df['label'].value_counts())
except Exception as e:
    print(f"Visualization failed: {e}")


try:
    # Display Sample Images
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    import random
    
    # Assuming df and TRAIN_PATH are defined in previous cells
    # TRAIN_PATH should be the path *inside* the Kaggle environment (e.g., /kaggle/input/.../train_images/)
    num_classes = df['label'].nunique()
    samples_per_class = 3
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 4, num_classes * 4))
    fig.suptitle('Sample Images for Each Cassava Disease Class', fontsize=16)
    
    for i, label in enumerate(sorted(df['label'].unique())):
        class_df = df[df['label'] == label]
        sample_images = random.sample(list(class_df['image_id']), samples_per_class)
        # Get class name if mapping exists, otherwise use label number
        # class_name = label_map.get(label, f'Class {label}') # Example if label map exists
        class_name = f'Class {label}' # Using label number as placeholder
    
        for j, image_id in enumerate(sample_images):
            try:
                img_path = os.path.join(TRAIN_PATH, image_id) # Use Kaggle path
                img = Image.open(img_path)
                ax = axes[i, j]
                ax.imshow(img)
                ax.set_title(f'{class_name}:\n{image_id}', fontsize=10)
                ax.axis('off')
            except FileNotFoundError:
                print(f"Warning: Image not found at {img_path}. Skipping.")
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.set_title(f'{class_name}:\n{image_id}', fontsize=10)
                ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
except Exception as e:
    print(f"Visualization failed: {e}")


df.info()


try:
    df.label.value_counts().plot(kind="bar")
except Exception as e:
    print(f"Visualization failed: {e}")


train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.1, random_state=42, stratify=df.label.values
)


class CassavaDataset(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path=DATA_PATH, mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        self.data_dir = "train_images" if mode == "train" else "test_images"

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(img)

        return image, label


# create image augmentations
transforms_train = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transforms_valid = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


print("Available Vision Transformer Models: ")
timm.list_models("vit*")


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
# if pretrained:
# self.model.load_state_dict(torch.load(MODEL_PATH))

            self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        ###################
        # train the model #
        ###################
        self.model.train()
        for i, (data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()
                # GPU/CPU handling only
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.int64)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.forward(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # Calculate Accuracy
                accuracy = (output.argmax(dim=1) == target).float().mean()
                # update training loss and accuracy
                epoch_loss += loss
                epoch_accuracy += accuracy

                # perform a single optimization step (parameter update)
                if False:  # TPU code removed


                    if i % 20 == 0:
                        print(f"\tBATCH {i+1}/{len(train_loader)} - LOSS: {loss}")

                    else:
                        optimizer.step()

                        return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)

    def validate_one_epoch(self, valid_loader, criterion, device):
        # keep track of validation loss
        valid_loss = 0.0
        valid_accuracy = 0.0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if device.type == "cuda":
                data, target = data.cuda(), target.cuda()
                # GPU/CPU handling only
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.int64)

                with torch.no_grad():
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = self.model(data)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # Calculate Accuracy
                    accuracy = (output.argmax(dim=1) == target).float().mean()
                    # update average validation loss and accuracy
                    valid_loss += loss
                    valid_accuracy += accuracy

                    return valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)


def fit_tpu(
    model, epochs, device, criterion, optimizer, train_loader, valid_loader=None
):

    valid_loss_min = np.Inf  # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in range(1, epochs + 1):
        gc.collect()
        para_train_loader = torch.utils.data.DataLoader(train_loader, [device])

        print(f"{'='*50}")
        print(f"EPOCH {epoch} - TRAINING...")
        train_loss, train_acc = model.train_one_epoch(
            para_train_loader.per_device_loader(device), criterion, optimizer, device
        )
        print(
            f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        gc.collect()

        if valid_loader is not None:
            gc.collect()
            para_valid_loader = torch.utils.data.DataLoader(valid_loader, [device])
            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc = model.validate_one_epoch(
                para_valid_loader.per_device_loader(device), criterion, device
            )
            print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            gc.collect()

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and epoch != 1:
                print(
                    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )


            valid_loss_min = valid_loss

    return {
        "train_loss": train_losses,
        "valid_losses": valid_losses,
        "train_acc": train_accs,
        "valid_acc": valid_accs,
    }


model = ViTBase16(n_classes=5, pretrained=True)


def _run():
    train_dataset = CassavaDataset(train_df, transforms=transforms_train)
    valid_dataset = CassavaDataset(valid_df, transforms=transforms_valid)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,


        shuffle=True,
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,


        shuffle=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=8,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        drop_last=True,
        num_workers=8,
    )

    criterion = nn.CrossEntropyLoss()
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    logs = fit_tpu(
        model=model,
        epochs=N_EPOCHS,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    print(f"Execution time: {datetime.now() - start_time}")

    print("Saving Model")
    # Corrected indentation for the line below
    torch.save(model.state_dict(), f'model_5e_{datetime.now().strftime("%Y%m%d-%H%M")}.pth')
# Removed extraneous closing parenthesis from the end of the function



