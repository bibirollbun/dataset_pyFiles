!pip install pydot
!apt-get install graphviz -y


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import math

rc = {
    "axes.facecolor": "#ffaaa5",
    "figure.facecolor": "#ffaaa5",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL


train = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
train.head().style.set_properties(**{'background-color':'green','color':'white','border-color':'#8b8c8c'})


test = pd.read_csv('/kaggle/input/UBC-OCEAN/test.csv')
test.head().style.set_properties(**{'background-color':'lightgreen','color':'black','border-color':'#8b8c8c'})


train.info()


# Summary statistics for relevant variables
styled_data = train.describe().style\
.background_gradient(cmap='summer')\
.set_properties(**{'text-align':'center','border':'1px solid black'})

# display styled data
display(styled_data)


# Class Distribution
class_distribution = train['label'].value_counts()
print(class_distribution)

# TMA Distribution
tma_distribution = train['is_tma'].value_counts()
print(tma_distribution)

# Correlation between Image Dimensions
correlation = train[['image_width', 'image_height']].corr()
print(correlation)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='image_width', y='image_height', data=train, hue='label')
plt.title('Scatter plot of Image Dimensions', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Scatter plot of Image Dimensions.png')
plt.show()



HGSC = train[train['label']=="HGSC"]
EC = train[train['label']=="EC"]
CC = train[train['label']=="CC"]
LGSC = train[train['label']=="LGSC"]
MC = train[train['label']=="MC"]

# set the figure size and font size
plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 14

# set the colors (I've selected a nice color scheme)
colors = ['#66b3ff','#99ff99','#ffcc99','#c2c2f0', '#ffb3e6']

# plot the pie chart for the training set
plt.subplot(1, 1, 1)
plt.pie([len(HGSC), len(EC), len(CC), len(LGSC), len(MC)], labels=['HGSC', 'EC', 'CC', 'LGSC', 'MC'], autopct='%1.1f%%', colors=colors)
plt.title('Training Set', fontsize = 12, fontweight = 'bold', color = 'darkred')

plt.suptitle('DistribuciÃ³n de subtipos de cÃ¡ncer de ovario', fontsize=14,fontweight = 'bold', color = 'darkgreen', y=1.05)

plt.savefig('Distribution of Subtypes of Ovarian Cancer.png')

# Show the plot
plt.show()



plt.figure(figsize=(12, 5))
sns.violinplot(x='label', y='image_width', data=train, inner='quartile')
plt.title('Violin Plot of Image Width by Label', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Violin Plot of Image Width by Label.png')
plt.show()

plt.figure(figsize=(12, 5))
sns.violinplot(x='label', y='image_height', data=train, inner='quartile')
plt.title('Violin Plot of Image Height by Label', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Violin Plot of Image Height by Label.png')
plt.show()




plt.figure(figsize=(12, 5))
sns.boxplot(x='is_tma', y='image_width', data=train)
plt.title('Box Plot of Image Width by TMA', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Box Plot of Image Width by TMA.png')
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x='is_tma', y='image_height', data=train)
plt.title('Box Plot of Image Height by TMA', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Box Plot of Image Height by TMA.png')
plt.show()



sns.pairplot(train[['image_width', 'image_height']])
plt.suptitle('Pairplot of Image Dimensions', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Pairplot of Image Dimensions.png')
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.savefig('Correlation Heatmap.png')
plt.show()



plt.figure(figsize=(12, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Class Distribution', fontsize=14, fontweight='bold', color='darkgreen')
plt.xlabel('Class Label', fontsize=12, fontweight='bold', color='darkblue')
plt.ylabel('Count', fontsize=12, fontweight='bold', color='darkblue')
plt.savefig('Class Distribution.png')
plt.show()


import glob
from matplotlib import pyplot as plt
from matplotlib.image import imread

# Define the paths to the image directories thumbnails
train_data = glob.glob('/kaggle/input/UBC-OCEAN/train_thumbnails/*.png')
test_data = glob.glob('/kaggle/input/UBC-OCEAN/test_thumbnails/*.png')

# Display a few sample images from the training set
num_samples = 5

fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

for i, image_path in enumerate(train_data[:num_samples]):
    img = imread(image_path)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f'Train Image {i+1}')

plt.tight_layout()
plt.savefig('Train Iamge.png')
plt.show()

# Display the one image from the testing set
fig, axes = plt.subplots(1, 1, figsize=(5, 5))  # Only one plot

# Check if there's at least one image in the test set
if len(test_data) > 0:
    img = imread(test_data[0])
    axes.imshow(img)
    axes.axis('off')
    axes.set_title('Test Image 1')

plt.tight_layout()
plt.savefig('Test Image.png')
plt.show()


import os
import matplotlib.pyplot as plt

# Count the number of images for each class in the training set
class_counts = {}
for image_path in train_data:
    class_name = os.path.basename(os.path.dirname(image_path))
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

# Create a bar plot for class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.title('Class Distribution in Training Set', fontsize = 14, fontweight = 'bold', color = 'darkgreen')
plt.xlabel('Class Label', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.ylabel('Count', fontsize = 12, fontweight = 'bold', color = 'darkblue')
plt.savefig('Class Distribution in Training Set.png')
plt.show()


from PIL import Image
import random

# Define the number of sample images to display
num_samples = 5

# Randomly select sample images from the training set
sample_images = random.sample(train_data, num_samples)

# Display the sample images
plt.figure(figsize=(15, 8))
for i, image_path in enumerate(sample_images, 1):
    image = Image.open(image_path)
    plt.subplot(1, num_samples, i)
    plt.imshow(image)
    plt.title(f'Sample {i}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('samples.png')
plt.show()


# Generate random training images (example)
num_samples = 1000
image_height = 28
image_width = 28
num_channels = 1  # For grayscale images

X_train = np.random.rand(num_samples, image_height, image_width, num_channels)

# Generate random labels (example)
num_classes = 10
y_train = np.random.randint(0, num_classes, size=num_samples)



# Generate random testing images (example)
num_samples_test = 200
X_test = np.random.rand(num_samples_test, image_height, image_width, num_channels)

# Generate random labels for testing (example)
y_test = np.random.randint(0, num_classes, size=num_samples_test)


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# One-hot encode the labels
y_train_encoded = to_categorical(y_train)

# Split the training data into training and validation sets
X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, validation_data=(X_val, y_val_encoded), epochs=5, batch_size=32)

# One-hot encode the labels for the test set (assuming y_test is defined)
y_test_encoded = to_categorical(y_test)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')

# Make predictions on the test set
predictions = model.predict(X_test)


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

img = plt.imread('model_architecture.png')
plt.imshow(img)
plt.axis('off') # Para ocultar los ejes
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import random

# Generate some sample true and predicted labels for demonstration purposes
y_true = np.random.randint(0, 3, size=100)
y_pred = np.random.randint(0, 3, size=100)

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 2. Classification Report
class_report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(class_report)


# The 'model' is your Sequential Keras model

# Sample Predictions with Images
sample_indices = random.sample(range(len(X_test)), 3)  # Get 3 random indices
sample_images = X_test[sample_indices]
sample_true_labels = y_test[sample_indices]

# Predict labels using probabilities
sample_pred_probs = model.predict(sample_images)
sample_pred_labels = np.argmax(sample_pred_probs, axis=1)

# Print the sample true and predicted labels
print("\nSample True Labels:")
print(sample_true_labels)

print("\nSample Predicted Labels:")
print(sample_pred_labels)

# Display sample images with true and predicted labels
plt.figure(figsize=(15, 5))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(sample_images[i])
    plt.title(f'True: {sample_true_labels[i]}, Predicted: {sample_pred_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample predict labels.png')
plt.show()


submission = pd.read_csv('/kaggle/input/UBC-OCEAN/sample_submission.csv')
submission.head().style.set_properties(**{'background-color':'blue','color':'white','border-color':'#8b8c8c'})


# Get the class with the highest probability for each sample
sample_pred_labels = np.argmax(sample_pred_probs, axis=1)

# Load the sample submission file
submission = pd.read_csv('/kaggle/input/UBC-OCEAN/sample_submission.csv')

# Update the 'label' column in the submission DataFrame
submission['label'] = sample_pred_labels[:len(submission)]

# Save the updated DataFrame as a CSV file
submission.to_csv('submission.csv', index=False)


submission

