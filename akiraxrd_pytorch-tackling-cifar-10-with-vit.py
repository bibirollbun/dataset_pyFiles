!pip install py7zr

# base
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm.notebook import tqdm

# machine learning
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler, Subset
from torchvision import datasets
from transformers import get_linear_schedule_with_warmup

# metric tracking
import time
from datetime import timedelta

# libraries for dataset unpacking / loading
import py7zr
import os
import PIL


if torch.cuda.is_available():
    print("CUDA is available; using as the main device")
    device = "cuda"
else:
    print("CUDA is not available; using CPU as the main device")
    device = "cpu"


vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
vit_transforms = vit_weights.transforms()

vit_transforms


labeled_dataset = datasets.CIFAR10(root="/kaggle/working",
                                   train=True,
                                   transform=vit_transforms,
                                   download=True)
classes = labeled_dataset.classes

image, label = next(iter(labeled_dataset))
print(f"Image tensor shape:  {image.shape}")
print(f"Label type: {type(label)}")


used_dataset_size = 10000
indices = torch.arange(used_dataset_size)

labeled_dataset = Subset(labeled_dataset, indices)

print(f"Used dataset size: {len(labeled_dataset)}")


train_dataset_size = int(0.8 * len(labeled_dataset))
validation_dataset_size = len(labeled_dataset) - train_dataset_size

train_dataset, validation_dataset = random_split(dataset=labeled_dataset,
                                                 lengths=(train_dataset_size, validation_dataset_size))

print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(validation_dataset)}")


batch_size = 16
train_sampler = RandomSampler(train_dataset)
validation_sampler = SequentialSampler(validation_dataset)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler)

validation_dataloader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   sampler=validation_sampler)

image_batch, label_batch = next(iter(train_dataloader))

sample_index = 3
image_sample = image_batch[sample_index]
label_sample = label_batch[sample_index]

plt.imshow(image_sample.permute(1, 2, 0))
plt.title(classes[label_sample])
plt.axis(False);


model = torchvision.models.vit_b_16(weights=vit_weights).to(device)

for parameter in model.parameters():
    parameter.requires_grad = False

model.heads = nn.Linear(768, len(classes)).to(device)


summary(model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])


epochs = 10
num_training_steps = len(train_dataloader) * epochs

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(),
                            lr=3e-3,
                            betas=(0.9, 0.999),
                            weight_decay=0.1)

schedule = get_linear_schedule_with_warmup(optimizer,
                                          num_warmup_steps=0,
                                          num_training_steps=num_training_steps)

optimizer


def format_time(elapsed_time_seconds: time.time) -> str:
    return str(timedelta(seconds=round(elapsed_time_seconds)))

def train_step(model,
               dataloader: DataLoader,
               loss_function: nn.Module,
               optimizer: torch.optim.Optimizer,
               schedule,
               device: torch.device) -> tuple[float, float]:
    """ Performs a single step on a batch from the given dataloader. Returns average loss and accuracy. """
    
    total_loss = 0.0
    total_accuracy = 0.0
    
    model.train()
    for feature_batch, label_batch in dataloader:
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)

        logits = model(feature_batch)
        loss = loss_function(logits, label_batch)
        total_loss += loss.item()

        predicted_classes = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        accuracy = (predicted_classes == label_batch).sum().item() / len(label_batch)
        total_accuracy += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy
    
def validation_step(model,
                   dataloader: DataLoader,
                   loss_function: nn.Module,
                   device: torch.device) -> tuple[float, float]:
    """ Evaluates the passed model on a single batch from the dataloader. Returns average loss and accuracy. """
    
    total_loss = 0.0
    total_accuracy = 0.0
    
    model.eval()
    with torch.inference_mode():
        for feature_batch, label_batch in dataloader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            
            logits = model(feature_batch)

            loss = loss_function(logits, label_batch)
            total_loss += loss.item()

            predicted_classes = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            accuracy = (predicted_classes == label_batch).sum().item() / len(label_batch)
            total_accuracy += accuracy

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy

def fit(model,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        schedule,
        epochs: int,
        device: torch.device):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }
    
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_start_time = time.time()
        
        train_loss, train_accuracy = train_step(model,
                                                dataloader=train_dataloader,
                                                loss_function=loss_function,
                                                optimizer=optimizer,
                                                schedule=schedule,
                                                device=device)

        validation_loss, validation_accuracy = validation_step(model,
                                                               dataloader=validation_dataloader,
                                                               loss_function=loss_function,
                                                               device=device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        
        #print(f"Epoch {epoch:2d}/{epochs} | train loss: {train_loss:.4f} | train accuracy: {train_accuracy:.4f} | validation loss: {validation_loss:.4f} | validation accuracy: {validation_accuracy:.4f}")
        print(f"Epoch {epoch:2d}/{epochs}")
        print(f"├ Training loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}")
        print(f"├ Validation loss: {validation_loss:.4f}, accuracy: {validation_accuracy:.4f}")
        print(f"└ Epoch took: {format_time(time.time() - epoch_start_time)}\n")

    return history


training_start_time = time.time()
history = fit(model,
              train_dataloader=train_dataloader,
              validation_dataloader=validation_dataloader,
              loss_function=loss_function,
              optimizer=optimizer,
              schedule=schedule,
              epochs=epochs,
              device=device)

print(f"Training took: {format_time(time.time() - training_start_time)}")


figure, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history["train_loss"], color="green", label="train_loss")
axes[0].plot(history["validation_loss"], color="orange", label="validation_loss")
axes[0].set(title="Train and Validation losses")
axes[0].legend()

axes[1].plot(history["train_accuracy"], color="green", label="train_accuracy")
axes[1].plot(history["validation_accuracy"], color="orange", label="validation_accuracy")
axes[1].set(title="Train and Validation accuracies")
axes[1].legend()


!ls /kaggle/input/cifar-10


image_dir = "/kaggle/working"

if not os.path.isdir(image_dir + "/test"):
    with py7zr.SevenZipFile(f"/kaggle/input/cifar-10/test.7z", mode="r") as zipf:
        zipf.extractall(image_dir)
    print(f"Test split zipfile extracted")

else:     
    print(f"Test split zipfile already extracted, skipping...")


submission = pd.read_csv("/kaggle/input/cifar-10/sampleSubmission.csv")
submission


class CIFAR10TestDataset(Dataset):
    def __init__(self,
                 image_folder: str,
                 image_ids: list,
                 transforms = None):
        self.image_folder = image_folder
        self.image_ids = image_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_ids[idx] + ".png")
        image = PIL.Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image


test_folder = "/kaggle/working/test"
test_image_ids = submission["id"].astype(str).to_list()
test_dataset = CIFAR10TestDataset(image_folder=test_folder,
                                  image_ids=test_image_ids,
                                  transforms=vit_transforms)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             sampler=test_sampler)


predicted_labels = []

model.eval()

for image_batch in tqdm(test_dataloader):
    image_batch = image_batch.to(device)

    with torch.inference_mode():
        logits = model(image_batch)

    logits = logits.detach().cpu()
    predicted_label_batch = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    predicted_labels.extend(predicted_label_batch.tolist())

predicted_classes = [classes[label] for label in predicted_labels]


submission["label"] = predicted_classes
submission.to_csv("submission.csv", index=False)

