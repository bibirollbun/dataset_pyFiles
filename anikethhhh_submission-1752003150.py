import os
import subprocess

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 


# It's really important to add an accelerator to your notebook, as otherwise the submission will fail.
# We recomment using the P100 GPU rather than T4 as it's faster and will increase the chances of passing the time cut-off threshold.

if DEVICE != 'cuda':
    raise RuntimeError('Make sure you have added an accelerator to your notebook; the submission will fail otherwise!')


# Helper functions for loading the hidden dataset.

def load_example(df_row):
    image = torchvision.io.read_image(df_row['image_path'])
    result = {
        'image': image,
        'image_id': df_row['image_id'],
        'age_group': df_row['age_group'],
        'age': df_row['age'],
        'person_id': df_row['person_id']
    }
    return result


class HiddenDataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, split='train'):
        super().__init__()
        self.examples = []

        df = pd.read_csv(f'/kaggle/input/neurips-2023-machine-unlearning/{split}.csv')
        df['image_path'] = df['image_id'].apply(
            lambda x: os.path.join('/kaggle/input/neurips-2023-machine-unlearning/', 'images', x.split('-')[0], x.split('-')[1] + '.png'))
        df = df.sort_values(by='image_path')
        df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
        if len(self.examples) == 0:
            raise ValueError('No examples.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example['image']
        image = image.to(torch.float32)
        example['image'] = image
        return example


def get_dataset(batch_size):
    '''Get the dataset.'''
    retain_ds = HiddenDataset(split='retain')
    forget_ds = HiddenDataset(split='forget')
    val_ds = HiddenDataset(split='validation')

    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return retain_loader, forget_loader, validation_loader


def unlearning(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader):
    """Generated from adversarial_forget_method implementation."""
    """Two-phase unlearning process implementation
    
    Args:
        net: The model to be unlearned
        retain_loader: DataLoader for retained training data
        forget_loader: DataLoader for data to be forgotten
        val_loader: DataLoader for validation data
        
    Returns:
        The unlearned model
    """
    # Phase 1: Adversarial forgetting
    adversarial_epochs = 1
    adversarial_criterion = nn.CrossEntropyLoss()
    adversarial_optimizer = optim.SGD(net.parameters(), lr=0.001,
                                      momentum=0.9, weight_decay=5e-4)

    for ep in range(adversarial_epochs):
        net.train()
        for batch_idx, sample in enumerate(forget_loader):
            if isinstance(sample, dict):
                inputs = sample["image"]
                targets = sample["age_group"]
            else:
                inputs, targets = sample  # For CIFAR format
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            adversarial_optimizer.zero_grad()
            outputs = net(inputs)
            loss = adversarial_criterion(outputs, targets)
            (-loss).backward()  # Gradient ascent
            adversarial_optimizer.step()

    # Phase 2: Utility restoration
    restore_epochs = 1
    restore_criterion = nn.CrossEntropyLoss()
    restore_optimizer = optim.SGD(net.parameters(), lr=0.001,
                                  momentum=0.9, weight_decay=5e-4)
    restore_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restore_optimizer, T_max=restore_epochs)

    for ep in range(restore_epochs):
        net.train()
        for batch_idx, sample in enumerate(retain_loader):
            if isinstance(sample, dict):
                inputs = sample["image"]
                targets = sample["age_group"]
            else:
                inputs, targets = sample  # For CIFAR format
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            restore_optimizer.zero_grad()
            outputs = net(inputs)
            loss = restore_criterion(outputs, targets)
            loss.backward()
            restore_optimizer.step()
        restore_scheduler.step()

    net.eval()
    return net



if os.path.exists('/kaggle/input/neurips-2023-machine-unlearning/empty.txt'):
    # mock submission
    subprocess.run('touch submission.zip', shell=True)
else:
    
    # Note: it's really important to create the unlearned checkpoints outside of the working directory 
    # as otherwise this notebook may fail due to running out of disk space.
    # The below code saves them in /kaggle/tmp to avoid that issue.
    
    os.makedirs('/kaggle/tmp', exist_ok=True)
    retain_loader, forget_loader, validation_loader = get_dataset(64)
    net = resnet18(weights=None, num_classes=10)
    net.to(DEVICE)
    for i in range(512):
        net.load_state_dict(torch.load('/kaggle/input/neurips-2023-machine-unlearning/original_model.pth'))
        unlearning(net, retain_loader, forget_loader, validation_loader)
        state = net.state_dict()
        torch.save(state, f'/kaggle/tmp/unlearned_checkpoint_{i}.pth')
        
    # Ensure that submission.zip will contain exactly 512 checkpoints 
    # (if this is not the case, an exception will be thrown).
    unlearned_ckpts = os.listdir('/kaggle/tmp')
    if len(unlearned_ckpts) != 512:
        raise RuntimeError('Expected exactly 512 checkpoints. The submission will throw an exception otherwise.')
        
    subprocess.run('zip submission.zip /kaggle/tmp/*.pth', shell=True)





