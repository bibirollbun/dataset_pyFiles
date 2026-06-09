import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score


# Custom Dataset Class to Load Data from CSV
class SheepDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path and load the image
        img_path = self.dataframe.iloc[idx]['filename']
        image = Image.open(img_path)

        # Get the corresponding label
        label = int(self.dataframe.iloc[idx]['label'])

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Initialize the tqdm progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)


        for inputs, labels in train_loader_tqdm:
            # Move data to the appropriate device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * inputs.size(0)

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Training complete')


# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    # Initialize the tqdm progress bar for validation
    val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=True)

    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, labels in val_loader_tqdm:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            # Get predictions (outputs are logits, apply softmax or argmax)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute average loss and accuracy
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
    return val_loss, val_accuracy


def display_images(model, image_paths, device, label_mapping, labell):
    # Reverse the label mapping for easier lookup of original labels
    rv_label_mapping = {v: k for k, v in label_mapping.items()}
    y_true  = []
    y_pred  = []
    # Define preprocessing transformation (works for both ResNet and VGG16)
    # Transformations
    transform = transforms.Compose([
    transforms.Resize((300, 300)),  
    transforms.CenterCrop(254),   
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  
                         [0.229, 0.224, 0.225])
    ])

    # Set up the figure for displaying images
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # Adjust grid size based on the number of images
    axes = axes.flatten()

    # Loop through each image and its corresponding label
    for i, image_path in enumerate(image_paths):
        try:
            image = Image.open(image_path)  # Load the image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue  # Skip to the next image if there's an error

        # Apply transformation to the image and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Set the model to evaluation mode
        model.eval()

        # Perform the prediction
        with torch.no_grad():
            output = model(image_tensor)  # Forward pass to get raw logits
            probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities

        # Get predicted class and the probability for that class
        _, predicted = torch.max(output, 1)  # This gives the index of the predicted class
        predicted_prob = probabilities[0, predicted.item()].item()  # Get the probability of the predicted class

        # Get predicted and original labels
        predicted_label = rv_label_mapping.get(predicted.item(), 'Unknown')
        original_label = rv_label_mapping.get(labell[i], 'Unknown')

        # Display the image with labels and predicted probability
        axes[i].imshow(image)
        axes[i].set_title(f'Predicted: {predicted_label} ({predicted_prob:.2f})\nOriginal: {original_label}')
        axes[i].axis('off')

        # Print for debugging (optional)
        #print(f"Predicted class for {image_path}: {predicted_label}, Probability: {predicted_prob:.4f}, Labeled Class: {original_label}")
        y_true .append(original_label)
        y_pred .append(predicted_label)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    #f1 score
    f1_output = f1_score(y_true,y_pred,average='macro') 
    print(f"F1 Score for the test data for model: {f1_output}")


import os
import pandas as pd
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F

def load_model_RES(model_save_path, label_mapping):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(label_mapping))  # 7 classes
    
    # Load the saved model state_dict
    state_dict = torch.load(model_save_path)
    
    # Remove the 'module.' prefix from the keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
        new_state_dict[name] = v

    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return model


if __name__ == '__main__':

    Folder = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/'

    Train_folder = Folder + "train/"
    Test_folder = Folder + "test/"
    CSV_data = pd.read_csv(os.path.join(Folder, 'train_labels.csv'))
    # Assuming label_mapping is the same dictionary you used for training
    label_mapping = {
        'Naeimi': 0,
        'Goat': 1,
        'Sawakni': 2,
        'Roman': 3,
        'Najdi': 4,
        'Harri': 5,
        'Barbari': 6
    }
    
    CSV_data['label'] = CSV_data['label'].map(label_mapping)
    # Convert image ID to the correct image path
    CSV_data['filename'] = CSV_data['filename'].apply(
        lambda x: os.path.join(Folder, 'train', x ))
    relevant_columns = ['filename', 'label']
    CSV_data = CSV_data[relevant_columns]

    # Split the data into train and test sets
    train_data, test_data = train_test_split(CSV_data, test_size=0.2, random_state=42)
    train_data = CSV_data
    # Transformations
    transform = transforms.Compose([
    transforms.Resize((300, 300)),  
    transforms.CenterCrop(254),   
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  
                         [0.229, 0.224, 0.225])
    ])
    
     # Create dataset and dataloader for training
    train_dataset = SheepDataset(dataframe=train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Create dataset and dataloader for validation
    test_dataset = SheepDataset(dataframe=test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Load the ResNet18 model
    if (False):
        
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 7)  # 7 classes 

    else:
        model_save_path ="/kaggle/input/sheep_resnet.pth/pytorch/default/1/trained_model_Sheep_ResNet18.pth"
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(label_mapping))  # 7 classes
        
        # Load the saved model state_dict
        state_dict = torch.load(model_save_path)
        
        # Remove the 'module.' prefix from the keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
            new_state_dict[name] = v
    
        # Load the new state_dict into the model
        model.load_state_dict(new_state_dict)
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    # Set up device, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Use DataParallel to wrap the model for multiple GPUs
        model = nn.DataParallel(model)
       
    except:
        pass
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=1000)

    # Validate the model
    validate_model(model, test_loader, criterion, device)

  
    # Select 20 random image paths and labels from the CSV
    random_samples = CSV_data.sample(n=20)
    random_image_paths = random_samples['filename'].tolist()
    labell = random_samples['label'].tolist()

    print("################RESNET################")
    # Display the output of 20 random images as titles in plot form
    display_images(model, random_image_paths, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), label_mapping, labell)
   
    # Save the model after training
    model_save_path = '/kaggle/working/trained_model_Sheep_ResNet18_2.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

  
    # test Model part
    test_image_folder = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/"

    # model eval mode
    model.eval()
    model.to(device)

    # Inverse label mapping (index → breed name)
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # ===== PREDICTION LOOP =====
    predictions = []
    
    for filename in tqdm(os.listdir(test_image_folder)):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(test_image_folder, filename)
    
            try:
                image = Image.open(image_path)
                input_tensor = transform(image).unsqueeze(0).to(device)
    
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = output.argmax(dim=1).item()
                    pred_label = inverse_label_mapping[pred_idx]
    
                predictions.append({
                    'filename': filename,
                    'label': pred_label
                })
    
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    # ===== SAVE TO CSV =====
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('/kaggle/working/submission4.csv', index=False)
    
    print("✅ Predictions saved to submission.csv")

