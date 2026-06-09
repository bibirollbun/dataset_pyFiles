import os
import pandas as pd
import shutil
import random

# Define paths
dataset_path = "/kaggle/input/isic-2024-challenge/train-image/image"  # Folder containing all images
output_path = "/kaggle/working/selected_images"  # Folder where selected images will be saved
metadata_path = "/kaggle/input/isic-2024-challenge/train-metadata.csv"  # Path to CSV file

# Load metadata
df = pd.read_csv(metadata_path)

# Separate class 1 and class 0 images
class_1_images = df[df['target'] == 1]['isic_id'].tolist()
class_0_images = df[df['target'] == 0]['isic_id'].tolist()

# Ensure all 300 class 1 images are included
selected_class_1 = list(class_1_images)  # Include all available class 1 images

# Randomly select 99,700 images from class 0
selected_class_0 = random.sample(class_0_images, 100000 - len(selected_class_1))

# Combine selected images
selected_images = selected_class_1 + selected_class_0

# Create output directory if not exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Copy selected images to the new folder
for image_id in selected_images:
    src_path = os.path.join(dataset_path, f"{image_id}.jpg")  # Assuming images are .jpg
    dest_path = os.path.join(output_path, f"{image_id}.jpg")
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
    else:
        print(f"Warning: {src_path} not found")

print("Dataset creation complete. Filtered dataset saved at:", output_path)


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_hair(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a rectangular structuring element
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    
    # Apply the black hat morphological operation
    b_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    
    # Threshold the black hat image to create a mask
    _, thresh = cv2.threshold(b_hat, 10, 255, cv2.THRESH_BINARY)
    
    # Use the inpainting technique to remove hair
    mod_image = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    
    return mod_image

def process_dataset(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct full file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Read the image
            image = cv2.imread(input_path)
            
            # Check if the image was successfully loaded
            if image is not None:
                # Remove hair from the image
                clean_image = remove_hair(image)
                
                # Save the cleaned image to the output folder
                cv2.imwrite(output_path, clean_image)
                
#                 print(f"Processed {filename}")
            else:
                print(f"Failed to load {filename}")

# Example usage: process all images in the dataset
input_folder = '/kaggle/working/selected_images/'  # Path to your input dataset
output_folder = '/kaggle/working/cleaned_images/'  # Path to save cleaned images

# Process the entire dataset
process_dataset(input_folder, output_folder)


import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Define the encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        enc = self.encoder(x)
        middle = self.middle(enc)
        dec = self.decoder(middle)
        return torch.sigmoid(dec)

# Load the U-Net model
model = UNet().to(device)
model.load_state_dict(torch.load('/kaggle/input/ham/pytorch/default/1/unet_skin_lesion_ham.pth', map_location=device), strict=False)
model.eval()

# Function to create a binary mask from an image
def create_mask(image_path):
    # Load the image
    original_image = cv2.imread(image_path)
    original_size = original_image.shape[1], original_image.shape[0]  # Width, Height

    # Resize and normalize the image for the model
    image_resized = cv2.resize(original_image, (256, 256))  # Resize as per the model's requirement
    image_normalized = image_resized / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Predict the mask
    with torch.no_grad():
        predicted_mask = model(image_tensor)

    # Convert predicted mask to NumPy array and binarize it
    predicted_mask = predicted_mask.squeeze(0).cpu().numpy()
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binarize the mask
    predicted_mask = predicted_mask[0]  # Select the first channel if it's multi-channel
    
    # Invert the mask: lesion area (1) -> black (0) and background (0) -> white (1)
    binary_mask = 1 - predicted_mask  # Invert mask (1 - predicted_mask)

    # Resize the binary mask to the original image size
    binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return binary_mask_resized

# Function to draw contours on the original image using the mask
def draw_contours(original_image, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image with reduced thickness
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Green contours with thickness of 1
    return contour_image

# Main function to process an image
def process_image(image_path):
    # Create a mask from the image
    mask = create_mask(image_path)

    # Load the original image
    original_image = cv2.imread(image_path)

    # Draw contours on the original image using the mask
    contour_image = draw_contours(original_image, mask)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title("Highlighted Lesion Contours")
    plt.axis('off')
    plt.show()

# Example usage
image_path = '/kaggle/input/isic-2024-challenge/train-image/image/ISIC_0052109.jpg'
process_image(image_path)


import os
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time
import h5py
import io
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# File paths
data_dir = '/kaggle/input/isic-2024-challenge/'
image_dir = '/kaggle/working/cleaned_images'

# Load metadata
metadata_path = os.path.join(data_dir, 'train-metadata.csv')
metadata_df = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv')
print(metadata_df.shape)


import os
import time
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Define paths
selected_images_folder = "/kaggle/working/selected_images"  # Use only the pre-selected dataset
metadata_path = "/kaggle/input/isic-2024-challenge/train-metadata.csv"

# Load metadata
df = pd.read_csv(metadata_path)

# Filter metadata to include only images in the selected dataset
available_images = {os.path.splitext(f)[0] for f in os.listdir(selected_images_folder)}
df_filtered = df[df['isic_id'].isin(available_images)].copy()

# Count class distribution
num_positive = df_filtered[df_filtered['target'] == 1].shape[0]
num_negative = df_filtered[df_filtered['target'] == 0].shape[0]
print(f"Filtered dataset contains {num_positive} positive and {num_negative} negative images.")

# Function to load images in parallel
def load_images_in_parallel(metadata_df, image_dir, num_workers=8):
    def load_image(row):
        isic_id = row['isic_id']
        label = row['target']
        image_path = os.path.join(image_dir, f"{isic_id}.jpg")
        
        try:
            img = Image.open(image_path)
            return img, label
        except FileNotFoundError:
            print(f"Warning: Image {image_path} not found!")
            return None, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, label

    images, labels = [], []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_image, row) for _, row in metadata_df.iterrows()]
        for future in futures:
            img, label = future.result()
            if img is not None:
                images.append(img)
                labels.append(label)

    return images, labels

# Use df_filtered as needed
print(df_filtered.head())  # Show the first few rows of the filtered dataframe

# Load images from the selected dataset
start_time = time.time()
loaded_images, labels = load_images_in_parallel(df_filtered, selected_images_folder, num_workers=8)
end_time = time.time()

print(f"Successfully loaded {len(loaded_images)} images in {end_time - start_time:.2f} seconds.")
print("Sample labels:", labels[:10])


# Function to display images
def display_images(images, title, num_images=4):
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=16)
    
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i][0])  # Display the image
        plt.axis('off')  # Hide axes
        plt.title('Malignant' if images[i][1] == 1 else 'Benign')  # Show class label as title

# Split loaded images into positive and negative classes
loaded_positive_images = [img for img in zip(loaded_images, labels) if img[1] == 1]  # Malignant cases
loaded_negative_images = [img for img in zip(loaded_images, labels) if img[1] == 0]  # Benign cases

# Display images from the positive class
display_images(loaded_positive_images, title='Malignant Cases', num_images=4)

# Display images from the negative class
display_images(loaded_negative_images, title='Benign Cases', num_images=4)

plt.show()


import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # For progress tracking

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        enc = self.encoder(x)
        middle = self.middle(enc)
        dec = self.decoder(middle)
        return torch.sigmoid(dec)

# Load the U-Net model
model = UNet().to(device)
model.load_state_dict(torch.load('/kaggle/input/ham/pytorch/default/1/unet_skin_lesion_ham.pth', map_location=device), strict=False)
model.eval()

# Function to create a binary mask from an image
def create_mask(image_path):
    original_image = cv2.imread(image_path)
    original_size = original_image.shape[1], original_image.shape[0]

    # Resize and normalize the image for the model
    image_resized = cv2.resize(original_image, (256, 256))
    image_normalized = image_resized / 255.0
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # Predict the mask
    with torch.no_grad():
        predicted_mask = model(image_tensor)

    predicted_mask = predicted_mask.squeeze(0).cpu().numpy()
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    predicted_mask = predicted_mask[0]
    
    # Invert the mask
    binary_mask = 1 - predicted_mask
    
    # Resize the binary mask to the original image size
    binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return binary_mask_resized

# Function to draw contours on the original image using the mask
def draw_contours(original_image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)  # Green contours with thickness of 1
    return contour_image

# Function to process all images in the dataset
def process_dataset(metadata_df, image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create class-specific folders
    positive_folder = os.path.join(output_dir, 'positive')
    negative_folder = os.path.join(output_dir, 'negative')
    os.makedirs(positive_folder, exist_ok=True)
    os.makedirs(negative_folder, exist_ok=True)

    # Process each image
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        isic_id = row['isic_id']
        label = row['target']  # 1 = malignant (positive), 0 = benign (negative)
        image_path = os.path.join(image_dir, f"{isic_id}.jpg")

        # Create a mask and draw contours on the original image
        mask = create_mask(image_path)
        original_image = cv2.imread(image_path)
        contour_image = draw_contours(original_image, mask)
        
        # Save the processed image in the appropriate folder
        if label == 1:
            output_path = os.path.join(positive_folder, f"{isic_id}_contour.jpg")
        else:
            output_path = os.path.join(negative_folder, f"{isic_id}_contour.jpg")
        
        cv2.imwrite(output_path, contour_image)

# Example usage
image_dir = '/kaggle/working/cleaned_images'
output_dir = '/kaggle/working/processed_images'
process_dataset(df_filtered, image_dir, output_dir)


import os
import matplotlib.pyplot as plt
import cv2

# Function to display a few processed images, # 1 = malignant (positive), 0 = benign (negative)
def display_processed_images(folder_path, num_images=5):
    processed_images = os.listdir(folder_path)[:num_images]  # Load first 'num_images' images
    plt.figure(figsize=(15, 15))
    
    for i, img_file in enumerate(processed_images):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Processed Image {i+1}")
        plt.axis('off')

    plt.show()

# Example: Display images from the 'positive' folder
print("Positive images: ")
display_processed_images('/kaggle/working/processed_images/positive', num_images=5)
print("Negative images: ")
display_processed_images('/kaggle/working/processed_images/positive', num_images=5)


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # Import joblib here

# Load the data from directories
def load_data(image_dir):
    images = []
    labels = []

    for label in ['positive', 'negative']:
        label_dir = os.path.join(image_dir, label)

        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                labels.append(1 if label == 'positive' else 0)

    return images, labels

# Extract contour features from the images
def extract_contour_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features.extend([area, perimeter])

    return features

# Prepare data for training
def prepare_data(images, labels):
    X = []
    y = []

    for image, label in zip(images, labels):
        contour_features = extract_contour_features(image)
        combined_features = contour_features
        X.append(combined_features)
        y.append(label)

    # Pad or truncate features to a fixed length
    max_length = max([len(features) for features in X])
    X = np.array([np.pad(features, (0, max_length - len(features))) for features in X])

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, np.array(y), scaler

# Train and evaluate the classifier
def train_and_evaluate(X_train, y_train, X_test, y_test, images_test, scaler):
    classifier = SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and scaler
    joblib.dump(classifier, "svm_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # Display some test images with their predictions
    plt.figure(figsize=(10, 5))
    for i in range(min(5, len(images_test))):  # Display up to 5 images
        plt.subplot(1, 5, i + 1)
        plt.imshow(cv2.cvtColor(images_test[i], cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Pred: {'Positive' if y_pred[i] == 1 else 'Negative'}")

    plt.show()

    return accuracy

# Main function
def main():
    image_dir = "/kaggle/working/processed_images"

    images, labels = load_data(image_dir)

    X, y, scaler = prepare_data(images, labels)

    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42
    )

    accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, images_test, scaler)

    print("Model Accuracy:", accuracy)

if __name__ == "__main__":
    main()


import os

# List files in the output_contour_images directory
output_dir = "/kaggle/working/processed_images"
print("Files in output contour images directory:", os.listdir(output_dir))


import os

# Define the paths for the negative and positive directories
negative_dir = "/kaggle/working/processed_images/negative"
positive_dir = "/kaggle/working/processed_images/positive"

# List files in the negative directory
# print("Files in negative directory:", os.listdir(negative_dir))

# List files in the positive directory
# print("Files in positive directory:", os.listdir(positive_dir))


import os

# Define the paths for the negative and positive directories
negative_dir = "/kaggle/working/processed_images/negative"
positive_dir = "/kaggle/working/processed_images/positive"

# Example file names
negative_file = 'ISIC_0553277.jpg'
positive_file = 'ISIC_0704481.jpg'  # Replace with an actual positive image filename

# Create absolute paths
negative_path = os.path.join(negative_dir, negative_file)
positive_path = os.path.join(positive_dir, positive_file)

print("Negative Image Path:", negative_path)
print("Positive Image Path:", positive_path)


if os.path.exists(negative_path):
    print(f"{negative_path} exists.")
else:
    print(f"{negative_path} does not exist.")



import cv2
import os
import numpy as np

# Function to extract contours from an image
def extract_contours(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to compare contours
def compare_contours(example_contours, class_contours):
    example_contour_hulls = [cv2.convexHull(cnt) for cnt in example_contours]
    similarities = []
    
    for class_contour in class_contours:
        class_contour_hulls = [cv2.convexHull(cnt) for cnt in class_contour]
        for example_hull in example_contour_hulls:
            for class_hull in class_contour_hulls:
                # Compare using shape matching
                match_value = cv2.matchShapes(example_hull, class_hull, cv2.CONTOURS_MATCH_I1, 0.0)
                similarities.append(match_value)

    return min(similarities)  # Return the smallest match value (best match)

# Load the example image
example_image_path = '/kaggle/working/processed_images/positive/ISIC_3450877_contour.jpg'  # Update with your actual image path
example_contours = extract_contours(example_image_path)

# Initialize a variable to hold the best match and the corresponding class
best_match_value = float('inf')
best_class = None

# Define the path to your processed image class directories
class_directories = [
    '/kaggle/working/processed_images/negative',
    '/kaggle/working/processed_images/positive'
]

# Compare with contours in each class
for class_dir in class_directories:
    class_contours = []
    
    # Load contours from each class directory
    for filename in os.listdir(class_dir):
        contour_path = os.path.join(class_dir, filename)
        if os.path.exists(contour_path):
            contours = extract_contours(contour_path)
            class_contours.append(contours)

    # Compare the example contours with class contours
    match_value = compare_contours(example_contours, class_contours)

    # Update best match if the current one is better
    if match_value < best_match_value:
        best_match_value = match_value
        best_class = class_dir

# Print the result
if best_class is not None:
    print(f"The example image belongs to class: {best_class} with match value: {best_match_value}")
else:
    print("No match found.")


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # Import joblib here

# Load the data from directories
def load_data(image_dir):
    images = []
    labels = []

    for label in ['positive', 'negative']:
        label_dir = os.path.join(image_dir, label)

        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                labels.append(1 if label == 'positive' else 0)

    return images, labels

# Extract contour features from the images
def extract_contour_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features.extend([area, perimeter])

    return features

# Prepare data for training
def prepare_data(images, labels):
    X = []
    y = []

    for image, label in zip(images, labels):
        contour_features = extract_contour_features(image)
        combined_features = contour_features
        X.append(combined_features)
        y.append(label)

    # Pad or truncate features to a fixed length
    max_length = max([len(features) for features in X])
    X = np.array([np.pad(features, (0, max_length - len(features))) for features in X])

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, np.array(y), scaler

# Train and evaluate the classifier
def train_and_evaluate(X_train, y_train, X_test, y_test, images_test, scaler):
    classifier = SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and scaler
    joblib.dump(classifier, "svm_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # Display some test images with their predictions
    
    plt.figure(figsize=(15, 10))  # Set figure size

    num_images = min(20, len(images_test))  # Ensure we don't exceed available images
    rows, cols = 4, 5  # Arrange in 4 rows × 5 columns

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)  # Correct subplot index
        plt.imshow(cv2.cvtColor(images_test[i], cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Pred: {'Positive' if y_pred[i] == 1 else 'Negative'}")

    plt.tight_layout()  # Adjust spacing
    plt.show()


    return accuracy

# Main function
def main():
    image_dir = "/kaggle/working/processed_images"

    images, labels = load_data(image_dir)

    X, y, scaler = prepare_data(images, labels)

    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42
    )

    accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, images_test, scaler)

    print("Model Accuracy:", accuracy)

if __name__ == "__main__":
    main()

