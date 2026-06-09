# ==============================
# CELL 1: INSTALL CLIP LIBRARY
# ==============================
# Install the CLIP library from its GitHub repository
!pip install git+https://github.com/openai/CLIP.git


# ==============================
# CELL 2: DEFINE PATH VARIABLES
# ==============================
# Set up the directories for training data, testing data, saving models, and saving predictions
training_directory = '/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/train'  # Path to training image directories
testing_directory = '/kaggle/input/vlg-recruitment-24-challenge/vlg-dataset/vlg-dataset/test'    # Path to testing images

hots_directory = '/kaggle/input/vlg-recruitment-24-challenge/Special-Package/Special-Package/final_examples' # Not for final model but for extra report

model_save_directory = '/kaggle/working/'                                            # Path to save the trained model
prediction_csv_path = '/kaggle/working/save_predictions(enhanced).csv'               # Name for predictions file


hots_prediction_csv_path = '/kaggle/working/save_hots_predictions.csv' 
hots_file_path = '/kaggle/input/vlg-recruitment-24-challenge/Special-Package/Special-Package/labels.txt'


# =============================================
# CELL 3: IMPORTING LIBRARIES AND DEVICE SETUP
# =============================================
# Import necessary Python libraries for file handling, data processing, and machine learning
import os  # For operating system dependent functionality
import glob  # For file pattern matching
import csv  # For reading and writing CSV files

import torch  # Main PyTorch library for tensor operations
import torchvision  # PyTorch library for computer vision
import torchvision.transforms as transforms  # For image transformations
from torch.utils.data import Dataset, DataLoader  # For handling datasets and loading data

from PIL import Image  # For image processing
import clip  # CLIP model Architecture for project
import numpy as np  # For numerical operations - mostly arrays

import torch.nn as nn  # For building neural network layers
import torch.optim as optim  # For optimization algorithms
from torch.optim.lr_scheduler import CyclicLR  # For learning rate scheduling

# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: " + device)



# =======================================================
# CELL 4: DEFINING ANIMAL CLASSES AND TEXT DESCRIPTIONS
# =======================================================

# List of 40 known animal classes that the model will be trained to recognize
known_animal_classes = [
    'antelope', 'bat', 'beaver', 'blue+whale', 'bobcat', 'buffalo', 'chihuahua', 'cow',
    'dalmatian', 'deer', 'dolphin', 'elephant', 'german+shepherd', 'giant+panda',
    'giraffe', 'grizzly+bear', 'hamster', 'hippopotamus', 'humpback+whale', 'killer+whale',
    'leopard', 'lion', 'mole', 'mouse', 'otter', 'ox', 'persian+cat', 'pig',
    'polar+bear', 'raccoon', 'rat', 'seal', 'siamese+cat', 'skunk', 'spider+monkey',
    'tiger', 'walrus', 'weasel', 'wolf', 'zebra'
]

# List of 10 additional animal classes that the model cannot see during training
unknown_animal_classes = [
    'horse', 'moose', 'gorilla', 'fox', 'sheep', 
    'chimpanzee', 'squirrel', 'rhinoceros', 'rabbit', 'collie'
]

# Textual prompts for each known class; helps text embeddings distinguish potential similarity between known and unknown classes
known_class_descriptions = {
    'antelope': 'A graceful, slender animal that can run incredibly fast and is found in Africa and Eurasia',
    'bat': 'A small, nocturnal mammal capable of sustained flight using echolocation',
    'beaver': 'A large, amphibious rodent known for building dams and lodges with branches and mud',
    'blue+whale': 'The largest animal on Earth, a massive marine mammal with a long, slender body',
    'bobcat': 'A medium-sized North American cat with spotted fur and short tail',
    'buffalo': 'A robust bovine species with large horns, often found in herds on grasslands',
    'chihuahua': 'A tiny dog breed with a lively personality and an apple-domed skull',
    'cow': 'A domesticated bovine widely raised for milk, meat and farm work',
    'dalmatian': 'A distinctively spotted, medium-sized dog breed known as a carriage or fire dog',
    'deer': 'A hoofed ruminant mammal of the family Cervidae, often with antlers in males',
    'dolphin': 'An intelligent marine mammal known for its playful behavior and high intelligence',
    'elephant': 'A massive herbivorous mammal with a trunk and ivory tusks, found in Africa and Asia',
    'german+shepherd': 'A large-sized breed of dog known for its intelligence and herding or guard abilities',
    'giant+panda': 'A black-and-white bear native to China that feeds mostly on bamboo',
    'giraffe': 'The tallest living terrestrial animal, easily recognized by its extremely long neck',
    'grizzly+bear': 'A large subspecies of brown bear with a muscular hump and strong forelimbs',
    'hamster': 'A small rodent often kept as a pet, known for storing food in its cheek pouches',
    'hippopotamus': 'A large, mostly herbivorous mammal with a barrel-shaped torso and enormous jaws',
    'humpback+whale': 'A large whale famous for its acrobatic behavior and complex whale songs',
    'killer+whale': 'Also known as orca, a highly social marine predator belonging to the dolphin family',
    'leopard': 'A big cat known for its spotted coat and climbing ability, widely distributed in Africa and Asia',
    'lion': 'A social big cat with males possessing a prominent mane, known as the king of the jungle',
    'mole': 'A small burrowing mammal with velvety fur and tiny eyes, adapted for underground life',
    'mouse': 'A small rodent with a pointed nose, furry round body, and a long tail',
    'otter': 'A playful, aquatic member of the weasel family with a streamlined body and webbed feet',
    'ox': 'A large, domesticated bovine used primarily for draft work in many parts of the world',
    'persian+cat': 'A long-haired cat breed characterized by its round face and shortened muzzle',
    'pig': 'A highly intelligent, domesticated omnivorous mammal with a snout used for foraging',
    'polar+bear': 'A large, white-furred bear living in the Arctic, highly adapted to cold climates',
    'raccoon': 'A medium-sized mammal with distinct facial mask markings and dexterous front paws',
    'rat': 'A rodent known for its adaptability, intelligence, and long, scaly tail',
    'seal': 'A marine mammal with a streamlined body, flippers, and typically whiskered face',
    'siamese+cat': 'A sleek, short-haired cat breed with distinctive point coloration and blue almond eyes',
    'skunk': 'A small to medium-sized mammal known for its ability to spray a foul-smelling liquid',
    'spider+monkey': 'A New World monkey with a prehensile tail and long limbs, known for agile movement in trees',
    'tiger': 'A powerful, striped big cat native to Asia and regarded as the largest cat species',
    'walrus': 'A large, flippered marine mammal with tusks, whiskers, and a bulky body',
    'weasel': 'A small, active predator with a slender body, known for its quick movements',
    'wolf': 'A wild canine with a pack-based social structure and a highly expressive face',
    'zebra': 'An African equid with distinctive black-and-white striped coats'
}

# Detailed descriptions for each unknown class to help the model recognize them without prior training - An attempt for Zero-Shot Training
unknown_class_descriptions = {
    'horse': 'A strong, four-legged mammal with a mane and tail, known for its speed and use by humans',
    'moose': 'A tall, long-legged animal with broad antlers, found in northern forests and wetlands',
    'gorilla': 'A powerful, predominantly herbivorous ape with a large body and gentle demeanor',
    'fox': 'A small to medium-sized omnivorous mammal with a bushy tail and elongated snout',
    'sheep': 'A wool-producing ruminant farm animal, often kept in flocks, used for meat and fiber',
    'chimpanzee': 'An intelligent ape closely related to humans, known for tool use and complex social groups',
    'squirrel': 'A small rodent with a bushy tail, known for climbing and storing nuts in trees',
    'rhinoceros': 'A huge herbivore with thick protective skin and one or two horns on its snout',
    'rabbit': 'A small mammal with long ears, known for rapid reproduction and hopping locomotion',
    'collie': 'An active, intelligent herding dog breed with a thick double coat and pointed snout'
}

# Combine class names with their descriptions to create text prompts for the CLIP model
text_descriptions = {}
for animal in known_animal_classes:
    # Replaces '+' with space for better readability in descriptions
    animal_name = animal.replace('+', ' ')
    # Create a detailed description for each known class
    text_descriptions[animal] = (
        f"A highly detailed professional photograph capturing the {animal_name}. "
        f"{known_class_descriptions[animal]}"
    )

for animal in unknown_animal_classes:
    # Replace '+' again
    animal_name = animal.replace('+', ' ')
    # Create a detailed description for each unknown class
    text_descriptions[animal] = (
        f"A highly detailed professional photograph capturing the {animal_name}. "
        f"{unknown_class_descriptions[animal]}"
    )

# Create a final list containing all 50 animal classes
all_animal_classes = known_animal_classes + unknown_animal_classes



# ================================================
# CELL 5: LOAD TRAINING DATASET FOR KNOWN CLASSES
# ================================================
# Define a custom dataset class for loading known animal images
class KnownAnimalsDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        # Initialize with the path to the training data and any image transformations
        self.transform = transform
        self.image_samples = []  # List to store image file paths and their labels
        self.class_list = known_animal_classes  

        # Loop through each class and collect file paths of each image
        for class_name in self.class_list:
            
            class_folder = os.path.join(folder_path, class_name)
            
            image_files = glob.glob(class_folder + '/*.*') # Using glob for easier file retrieval
            
            for image_file in image_files:
                if os.path.isfile(image_file):  # Checks for correct file type to prevent error
                   
                    self.image_samples.append((image_file, class_name)) # Creates the required image-label pair

    def __len__(self):
        # Return the total number of samples
        return len(self.image_samples)

    def __getitem__(self, index):
        # Get the image path and label for the given index
        image_path, label = self.image_samples[index]
        # Open the image and convert it to RGB format
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            # Apply required transformations to the image
            image = self.transform(image)
        # Return the transformed image and the index of its class
        return image, self.class_list.index(label)



# ====================================
# CELL 6: DEFINE DATA AUGMENTATIONS
# ====================================
# I apply various random transformations to training images to make the dataset diverse

image_transformations = transforms.Compose([
    # Randomly crop the image to 336x336 pixels with slight scale variations
    transforms.RandomResizedCrop((336,336), scale=(0.8, 1.0)),
    # Randomly flip the image horizontally with 50% chance
    transforms.RandomHorizontalFlip(p = 0.5),
    # Randomly change the brightness, contrast, saturation, and hue of the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # Randomly rotate the image by up to 15 degrees
    transforms.RandomRotation(degrees=15),
    # Randomly grayscale picture with 20% probability
    transforms.RandomGrayscale(p=0.2),
    # Convert the PIL image to a PyTorch tensor for uniform processing
    transforms.ToTensor(),

    
    # Randomly erase a part of the image with a 10% probability
    transforms.RandomApply([transforms.RandomErasing(p=1.0)], p=0.1),

    
    # Normalize the image tensor with mean and standard deviation values
    transforms.Normalize([0.485, 0.456, 0.406],  # Using ImageNet statistics
                         [0.229, 0.224, 0.225])
])

# Creates the dataset with the training directory and applied transformations for an augmented data
full_training_dataset = KnownAnimalsDataset(training_directory, transform=image_transformations)


# ====================================================
# CELL 7: SPLITTING DATASET INTO TRAIN AND VALIDATION
# ====================================================

# Define the size of the validation set (15% of the full training dataset)
validation_split = 0.15
validation_size = int(len(full_training_dataset) * validation_split)
training_size = len(full_training_dataset) - validation_size

# Splitting the dataset
training_dataset, validation_dataset = torch.utils.data.random_split(
    full_training_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(13)  # Manual Seed for reproducibility every time model is trained
)


# ==============================
# CELL 8: CREATING DATALOADERS 
# ==============================

# DataLoader for training data
training_loader = DataLoader(
    training_dataset,
    batch_size=32,  
    shuffle=True,    # Shuffles the data at every epoch
    num_workers=4    # Kaggle processing limit is 4
)

# DataLoader for validation data
validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,  
    shuffle=False,  # No need to shuffle validation data
    num_workers=4    
)



# ====================================
# CELL 9: LOAD AND PREPARE THE PRETRAINED CLIP MODEL
# ====================================
# Load the CLIP model and its preprocessing steps
clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)  

# Freeze all layers in the model except for specific ones to allow partial fine-tuning
for layer_name, layer_parameters in clip_model.named_parameters():
    # Only allow gradients for certain layers to fine-tune 
    if (
        "visual.transformer.resblocks" in layer_name
        or "visual.ln_post" in layer_name
        or "visual.proj" in layer_name
    ):
        layer_parameters.requires_grad = True  # These layers will be updated during training
    else:
        layer_parameters.requires_grad = False  # These layers will remain frozen

# Determine the size of the image embedding by passing a dummy image through the model
with torch.no_grad():
    dummy_image = torch.zeros(1, 3, 336, 336).to(device)  # Create a dummy image tensor
    dummy_features = clip_model.encode_image(dummy_image)  # Get image embeddings
image_embedding_size = dummy_features.shape[1]  

# Add a new classifier with additional layers to map image embeddings to the number of known classes


classifier_layer = nn.Sequential(
    nn.Linear(image_embedding_size, 512),         # First linear layer from embedding to 512 neurons
    nn.ReLU(),                                    # ReLU activation function for non-linearity
    nn.BatchNorm1d(512),                          # Batch normalization for stable training
    nn.Dropout(0.3),                              # Dropout layer to prevent overfitting
    nn.Linear(512, 256),                          # Second linear layer from 512 to 256 neurons
    nn.ReLU(),                                    # ReLU activation function
    nn.Dropout(0.3),                              # Additional dropout
    nn.Linear(256, len(known_animal_classes))    # Third linear layer to number of classes
).to(device)


# Define the optimizer to update the model's parameters during training

optimizer_parameters = [
    {"params": [param for name, param in clip_model.named_parameters() if param.requires_grad], "lr": 1e-5},
    {"params": classifier_layer.parameters(), "lr": 3e-4}
]
optimizer = optim.AdamW(optimizer_parameters, weight_decay=1e-4)


learning_rate_scheduler = CyclicLR(
    optimizer,
    base_lr=3e-5,  # Lower bound of the learning rate
    max_lr=3e-4,    # Upper bound of the learning rate
    step_size_up = 100,  # Number of iterations to increase the learning rate
    mode="triangular2",  # Shape of the learning rate cycle
    cycle_momentum=False  # Whether to cycle momentum
)

# Define the loss function to measure how well the model is performing
loss_function = nn.CrossEntropyLoss()



# ====================================
# CELL 10: TRAIN THE MODEL WITH VALIDATION AND EARLY STOPPING
# ====================================
# Import tqdm for progress bars to better visualise the processing part
from tqdm import tqdm

# Setting the epochs
number_of_epochs = 10  

# Parameters for Early Stopping
early_stopping_patience = 5  # Number of epochs to wait for improvement before stopping
best_validation_loss = float('inf')  # Initialize the best validation loss as infinity
epochs_without_improvement = 0  # Counter for epochs without improvement for early stopping

# Start the training loop
for current_epoch in range(number_of_epochs):
    # Setting the model and classifier to training mode
    clip_model.train()
    classifier_layer.train()
    
    total_training_loss = 0.0  # To calculate loss over the training epoch
    total_training_correct = 0  # To count correct predictions during training
    total_training_samples = 0  # To count total training samples processed

    # Use tqdm to display a progress bar for the training batches
    training_batches = tqdm(training_loader, desc=f"Epoch {current_epoch + 1}/{number_of_epochs} - Training", unit="batch")
    
    # Loop through each batch of data in the training loader
    for batch_images, batch_labels in training_batches:
        # Move images and labels to the selected device (GPU or CPU)
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # Encode the images with text to get their features using the CLIP model
        with torch.no_grad():
            image_features = clip_model.encode_image(batch_images)
            image_features = image_features.float()  

        # Pass the image features through the classifier to get predictions
        predictions = classifier_layer(image_features)
        
        # Calculate the loss between predictions and actual labels
        loss = loss_function(predictions, batch_labels)

        # Clear previous gradients
        optimizer.zero_grad()
        # Backpropagate the loss to compute gradients
        loss.backward()
        # Update the model parameters based on gradients
        optimizer.step()
        # Update the learning rate using the scheduler
        learning_rate_scheduler.step()

        # Accumulate the loss
        total_training_loss += loss.item() * batch_labels.size(0)
        # Get the predicted classes by selecting the highest probability
        _, predicted_classes = torch.max(predictions.detach(), 1)
        
        # Count how many predictions were correct
        total_training_correct += (predicted_classes == batch_labels).sum().item()
        # Update the total number of samples processed
        total_training_samples += batch_labels.size(0)
        
        # Calculate current loss and accuracy for the batch
        current_training_loss = total_training_loss / total_training_samples
        current_training_accuracy = 100.0 * total_training_correct / total_training_samples
        # Update the tqdm progress bar with current loss and accuracy
        training_batches.set_postfix(loss=current_training_loss, accuracy=f"{current_training_accuracy:.2f}%")

    # Calculate the average loss and accuracy for the training epoch
    average_training_loss = total_training_loss / total_training_samples
    training_accuracy_percentage = 100.0 * total_training_correct / total_training_samples

###########################################################################################################################
    
    # Set the model and classifier to evaluation mode
    clip_model.eval()
    classifier_layer.eval()
    
    total_validation_loss = 0.0  # To calculate loss over the validation epoch
    total_validation_correct = 0  # To count correct predictions during validation
    total_validation_samples = 0  # To count total validation samples processed

    
    validation_batches = tqdm(validation_loader, desc=f"Epoch {current_epoch + 1}/{number_of_epochs} - Validation", unit="batch")
    
    # Disable gradient calculations for validation to speed up computation
    with torch.no_grad():
        # Loop through each batch of data in the validation loader
        for val_images, val_labels in validation_batches:
            # Move images and labels to the selected device (GPU or CPU)
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            # Encode the images to get their features using the CLIP model
            image_features = clip_model.encode_image(val_images)
            image_features = image_features.float()  

            # Pass the image features through the classifier to get predictions
            val_predictions = classifier_layer(image_features)
            
            # Calculate the loss between predictions and actual labels
            val_loss = loss_function(val_predictions, val_labels)

            # Accumulate the loss
            total_validation_loss += val_loss.item() * val_labels.size(0)

            
            # Get the predicted classes by selecting the highest probability
            _, val_predicted_classes = torch.max(val_predictions, 1)
            # Count how many predictions were correct
            total_validation_correct += (val_predicted_classes == val_labels).sum().item()
            # Update the total number of samples processed
            total_validation_samples += val_labels.size(0)
            
            # Calculate current loss and accuracy for the batch
            current_validation_loss = total_validation_loss / total_validation_samples
            current_validation_accuracy = 100.0 * total_validation_correct / total_validation_samples
            # Update the tqdm progress bar with current loss and accuracy
            validation_batches.set_postfix(loss=current_validation_loss, accuracy=f"{current_validation_accuracy:.2f}%")
    
    # Calculate the average loss and accuracy for the validation epoch
    average_validation_loss = total_validation_loss / total_validation_samples
    validation_accuracy_percentage = 100.0 * total_validation_correct / total_validation_samples

   
    # Check if the validation loss has improved
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss  # Update the best validation loss
        epochs_without_improvement = 0  # Reset the counter
        # Save the best model weights
        torch.save({
            'clip_model_weights': clip_model.state_dict(),
            'classifier_weights': classifier_layer.state_dict()
        }, os.path.join(model_save_directory, 'best_clip_finetuned.pth'))
        print(f"Validation loss improved to {best_validation_loss:.4f}. Model saved.")
    else:
        epochs_without_improvement += 1  # Increase the counter
        print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
        # Check if model reached the patience limit
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered. Training halted.")
            break  # Exit the training loop

    # Printing the results for the current epoch
    print(f"Epoch [{current_epoch + 1}/{number_of_epochs}], "
          f"Training Loss: {average_training_loss:.4f}, Training Accuracy: {training_accuracy_percentage:.2f}%, "
          f"Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy_percentage:.2f}%")



# ====================================
# CELL 11: SAVE THE TRAINED MODEL
# ====================================
# Check if the save directory exists; if not, create it
if not os.path.exists(model_save_directory):
    os.makedirs(model_save_directory, exist_ok=True)

# Save the state dictionaries of both the CLIP model and the classifier layer
torch.save({
    'clip_model_weights': clip_model.state_dict(),
    'classifier_weights': classifier_layer.state_dict()
}, os.path.join(model_save_directory, 'clip_finetuned_final.pth'))


print("Training Completed, Model saved succesfully")



# ====================================
# CELL 12: TEST THE MODEL AND MAKE PREDICTIONS
# ====================================
# Precompute the text embeddings for all 50 classes using CLIP's text encoder

# Tokenize all text prompts and move them to the selected device
text_tokens = torch.cat([clip.tokenize(text_descriptions[class_name]) for class_name in all_animal_classes]).to(device)

# Encode the text prompts to get their embeddings
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    # Normalize the text embeddings to have unit length
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Define a custom dataset class again but for loading test images this time
class TestImagesDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        # Initialize with the path to the test data and any image transformations
        self.transform = transform
        # Get all image file paths in the test directory
        self.image_files = sorted(glob.glob(os.path.join(folder_path, '*.*')))
        # Extract image filenames from the file paths
        self.image_names = [os.path.basename(file_path) for file_path in self.image_files]

    def __len__(self):
        # Return the total number of test images
        return len(self.image_files)

    def __getitem__(self, index):
        # Get the image path and name for the given index
        image_path = self.image_files[index]
        image_name = self.image_names[index]
        # Open the image and convert it to RGB format
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            # Apply any transformations to the image
            image = self.transform(image)
        # Return the transformed image and its filename
        return image, image_name

# Define transformations for test images to match the preprocessing used during training
# No Random augmentations used however 
test_image_transformations = transforms.Compose([
    # Resize the image to 336 x 336 pixels
    transforms.Resize((336, 336)),
    # Convert the PIL image to a PyTorch tensor
    transforms.ToTensor(),
    # Normalize the image tensor with mean and standard deviation values
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # Same as training normalization
        std=(0.229, 0.224, 0.225)
    )
])

# Create an instance of the test dataset with the testing directory and transformations
test_dataset = TestImagesDataset(testing_directory, transform=test_image_transformations)




test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Set the model and classifier to evaluation mode 
clip_model.eval()
classifier_layer.eval()

# Initialize a list to store predictions
all_predictions = []

# Disable gradient calculations for faster processing
with torch.no_grad():
    # Loop through each batch of test data
    for test_images, test_image_names in tqdm(test_loader, desc="Testing", unit="batch"):
        # Move test images to the selected device (GPU or CPU)
        test_images = test_images.to(device)

        # Encode the test images to get their features using the CLIP model
        image_features = clip_model.encode_image(test_images)
        # Normalize the image embeddings to have unit length
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity between image features and all text embeddings
        # This results in a similarity score for each class per image
        similarity_scores = image_features @ text_features.T  # Matrix multiplication

        # Apply softmax to get probabilities for each class
        probability_scores = similarity_scores.softmax(dim=-1)
        # Get the index of the class with the highest probability for each image
        top_class_indices = probability_scores.argmax(dim=-1).cpu().numpy()

        # Loop through each prediction and store the results
        for i, class_index in enumerate(top_class_indices):
            predicted_class = all_animal_classes[class_index]  # Get class name from index
            all_predictions.append((test_image_names[i], predicted_class))  # Append to predictions list

print("Testing Complete")



# ====================================
# CELL 13: SAVE PREDICTIONS TO CSV FILE
# ====================================
# Sort the predictions by image filename for consistency
all_predictions.sort(key=lambda x: x[0])

# Open the CSV file in write mode
with open(prediction_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(['image_id', 'class'])
    # Write each prediction as a new row in the CSV
    for image_name, predicted_class in all_predictions:
        csv_writer.writerow([image_name, predicted_class])

print("The predictions are now saved in 'save_predictions.csv' in the specified directory")



# =============================================
# CELL 14: HOTS SAMPLE TESTING WITH SAME CODE
# =============================================

# Create an instance of the test dataset with the testing directory and transformations
test_dataset = TestImagesDataset(hots_directory, transform=test_image_transformations)



test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Set the model and classifier to evaluation mode 
clip_model.eval()
classifier_layer.eval()

# Initialize a list to store predictions
all_hots_predictions = []

# Disable gradient calculations for faster processing
with torch.no_grad():
    # Loop through each batch of test data
    for test_images, test_image_names in tqdm(test_loader, desc="Testing HOTS", unit="batch"):
        # Move test images to the selected device (GPU or CPU)
        test_images = test_images.to(device)

        # Encode the test images to get their features using the CLIP model
        image_features = clip_model.encode_image(test_images)
        # Normalize the image embeddings to have unit length
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity between image features and all text embeddings
        # This results in a similarity score for each class per image
        similarity_scores = image_features @ text_features.T  # Matrix multiplication

        # Apply softmax to get probabilities for each class
        probability_scores = similarity_scores.softmax(dim=-1)
        # Get the index of the class with the highest probability for each image
        top_class_indices = probability_scores.argmax(dim=-1).cpu().numpy()

        # Loop through each prediction and store the results
        for i, class_index in enumerate(top_class_indices):
            predicted_class = all_animal_classes[class_index]  # Get class name from index
            all_hots_predictions.append((test_image_names[i], predicted_class))  # Append to predictions list

print("Testing Complete")



# ======================================
# CELL 15: SAVE PREDICTIONS TO CSV FILE
# ======================================
# Sort the predictions by image filename for consistency
all_hots_predictions.sort(key=lambda x: x[0])

# Open the CSV file in write mode
with open(hots_prediction_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(['image_id', 'class'])
    # Write each prediction as a new row in the CSV
    for image_name, predicted_class in all_hots_predictions:
        csv_writer.writerow([image_name, predicted_class])

print("The predictions are now saved in the specified directory")



# =================================
# CELL 16: CHECKING HOTS ACCURACY
# =================================

# Open the txt file and read each line, remove any extra spaces, and append to the list
with open(hots_file_path, mode='r') as label_file:
    accurate_result = []
    for line in label_file:
        accurate_result.append(line.strip())

counter = 0

# Compare each value in all_hots_predictions with the accurate_result list
for index in range(len(accurate_result)):
    if all_hots_predictions[index][1] == accurate_result[index]:
        counter += 1

# Calculate accuracy as a percentage
accuracy = (counter / len(accurate_result)) * 100

# Print the accuracy result
print("Accuracy for HOTS: ", accuracy, "%")


