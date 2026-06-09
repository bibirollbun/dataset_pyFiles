import h5py
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler, SGD
import time
from tempfile import TemporaryDirectory
import pandas as pd


# Load and display (x,y) spot locations and cell type annotation table for Train slides
with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    train_spots = f["spots/Train"]
    
    # Dictionary to store DataFrames for each slide
    train_spot_tables = {}
    
    for slide_name in train_spots.keys():
        # Load dataset as NumPy structured array
        spot_array = np.array(train_spots[slide_name])
        
        # Convert to DataFrame
        df = pd.DataFrame(spot_array)
        
        # Store in dictionary
        train_spot_tables[slide_name] = df

# Example: Display the spots table for slide 'S_1'
train_spot_tables['S_1']


# Display spot table for Test slide (only the spot coordinates on 2D array)
with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    test_spots = f["spots/Test"]
    spot_array = np.array(test_spots['S_7'])
    test_spot_table = pd.DataFrame(spot_array)
    
# Show the test spots coordinates for slide 'S_7'
test_spot_table


hyper_params = {
    "patch_size": 224,
    "batch_size": 32,
    "lr": 0.001,
    "gamma": 0.1,
    "epochs": 12,
    "step": 7,
}


# Define image transformations for training, validation, and test sets
transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

# Custom dataset class to extract and process patches from HDF5 slides
class CustomDataset(Dataset):
    def __init__(self, data_path, slide_names, transform=None, is_train=True):
        self.transform = transform
        self.data = []
        self.targets = []
        self.patch_size = hyper_params["patch_size"]
        self.slide_names = slide_names
        self.is_train = is_train

        with h5py.File(data_path, "r") as f:
            images = f["images/Train"] if self.is_train else f["images/Test"]
            coords = f["spots/Train"] if self.is_train else f["spots/Test"]
            for slide_name in self.slide_names:
                slide = np.array(images[slide_name])
                spots = np.array(coords[slide_name])
                df = pd.DataFrame(spots)

                # Apply specific x/y shifts for alignment if needed
                x_shift, y_shift = 0, 0
                if slide_name == 'S_1':
                    x_shift, y_shift = 50, 60
                elif slide_name == 'S_2':
                    x_shift, y_shift = 95, 55
                df['x'] -= x_shift
                df['y'] -= y_shift

                # Extract patches and corresponding targets
                for _, row in df.iterrows():
                    x_center, y_center = int(row['x']), int(row['y'])
                    x0 = x_center - self.patch_size // 2
                    y0 = y_center - self.patch_size // 2
                    patch = slide[y0:y0 + self.patch_size, x0:x0 + self.patch_size, :]
                    self.data.append(patch)
                    self.targets.append(row[2:].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.targets[index]
        data = self.data[index]
        if self.transform:
            return (self.transform(data), target) if self.is_train else self.transform(data)
        else:
            return (data, target) if self.is_train else data

# Wrapper to apply transforms to dataset subsets (e.g., train/val split)
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.subset[index]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset)

# Dataset paths and slide setup
data_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
batch_size = hyper_params["batch_size"]
train_slides = ['S_1', 'S_2', 'S_3', 'S_4', 'S_5']
val_slide = ['S_6']
test_slide = ['S_7']

# Load full dataset and split into train/val subsets
dataset = CustomDataset(data_path, slide_names=train_slides + val_slide, transform=None)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, val_size])

# Wrap subsets with respective transforms
train_dataset = TransformedSubset(train_subset, transform['train'])
val_dataset = TransformedSubset(val_subset, transform['val'])
test_dataset = CustomDataset(data_path, slide_names=test_slide, is_train=False, transform=transform['test'])

# Create final dataset and dataloaders
dataset = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
}
dataloaders = {
    x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val']
}
test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val', 'test']}

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



# Computes the average Spearman correlation coefficient row-wise
# between the true and predicted values
def rowwise_spearman(y_true, y_pred):
    scores = []
    for yt, yp in zip(y_true, y_pred):
        coef, _ = spearmanr(yt, yp)  # Compute Spearman correlation for each row
        scores.append(coef)
    return np.mean(scores)  # Return average correlation across all rows


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_score = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_score = 0.0
                # Iterate over data.
                for inputs, targets in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device).float()
                    targets = targets.to(device).float()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        score = rowwise_spearman(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_score += score * inputs.size(0)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_score = running_score / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Score: {epoch_score:.4f}')
                
                # deep copy the model
                if phase == 'val' and epoch_score > best_score:
                    best_score = epoch_score
                    torch.save(model.state_dict(), best_model_params_path)

            print()
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Score: {best_score:4f}')
        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


# Custom transfer learning model using a pretrained CNN as a frozen feature extractor
class TransferModel(nn.Module):
    def __init__(self, pretrained):
        super(TransferModel, self).__init__()
        
        # Extract all layers except the classification head from the pretrained model
        self.feature_extractor = nn.Sequential(*list(pretrained.children())[:-1])
        
        # Custom head: reduces channels, flattens, then maps to 35 output features
        self.custom_layers = nn.Sequential(
            nn.Conv2d(1792, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256, 35)
        )

        # Freeze pretrained model parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Enable training only on the custom head
        for param in self.custom_layers.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.custom_layers(x)
        return x



# Set random seed for reproducibility
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 123
seed_everything(seed)

# Ensure deterministic behavior in cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Set up training parameters and model
epochs = hyper_params["epochs"]
lr = hyper_params["lr"]
gamma = hyper_params["gamma"]
step = hyper_params["step"]

# Load EfficientNet-B4 pretrained weights and initialize custom model
pretrained = models.efficientnet_b4(pretrained=True)
model = TransferModel(pretrained)
model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

# Train the model
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)


# Run model in evaluation mode on test set and collect predictions
was_training = model.training  # Save current training state
model.eval()  # Switch to evaluation mode (disables dropout, etc.)

preds = []
with torch.no_grad():  # Disable gradient computation for inference
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds.append(outputs.cpu().numpy())  # Move predictions to CPU

model.train(mode=was_training)  # Restore original training state

# Concatenate predictions from all batches
preds = np.concatenate(preds)


# Create a random submission
# (predictions of cell type abundances for 35 classes across the Test slide spots;
# spot order should be same as in the 'Test' spots table)

# Use the cell type columns from the train spots table; assuming first two columns are (x, y)
cell_type_columns = train_spot_tables['S_1'].columns[2:].values  # Expecting 35 cell types here
indices = test_spot_table.index.values  # All spots on the Test slide

# Create a 2D array of random floats between 0 and 2 for each spot and cell type
#prediction_matrix = 2 * np.random.rand(len(indices), len(cell_type_columns))
predicted_labels = pd.DataFrame(preds, columns=cell_type_columns, index=indices)

predicted_labels.head()


# Prepare submission DataFrame: spot_id column and then predictions for each cell type
submission_df = predicted_labels.copy()
submission_df.insert(0, 'ID', submission_df.index)

# Save the submission file as submission.csv
submission_df.to_csv("./submission.csv", index=False)
print("Submission file 'submission.csv' created!")

