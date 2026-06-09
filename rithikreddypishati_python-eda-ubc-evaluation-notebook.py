

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from IPython.display import HTML




train_csv_path = "/kaggle/input/UBC-OCEAN/train.csv"
train_images_path = "/kaggle/input/UBC-OCEAN/train_images"
train_thumbnails_path = "/kaggle/input/UBC-OCEAN/train_thumbnails"
test_csv_path = "/kaggle/input/UBC-OCEAN/test.csv"
test_images_path = "/kaggle/input/UBC-OCEAN/test_images"
test_thumbnails_path = "/kaggle/input/UBC-OCEAN/test_images"


df = pd.read_csv(train_csv_path)



df.head()






# Verify dataset size
print("Size of the dataset: ", df.shape[0])
assert df.shape[0] > 0, "Dataset size must be greater than 0"


def plot_images(folder_path: str, resize: bool = False):
    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    num_images_to_plot = 6
    selected_images = random.sample(image_files, num_images_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_images_to_plot:
            image_path = os.path.join(folder_path, selected_images[i])
            img = Image.open(image_path)
            if resize:
                img = img.resize((512,512))
            img = np.array(img)
            ax.imshow(img)
            ax.set_title(selected_images[i])
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    

# Test the function
plot_images('/kaggle/input/UBC-OCEAN/train_thumbnails', resize=True)






label_df = pd.DataFrame(df['label'].value_counts())
label_df.reset_index(inplace=True)
plt.bar(label_df['label'], label_df['count'], color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.show()
assert len(label_df) > 0, "Label distribution must be displayed correctly"


HGSC = df[df['label']=="HGSC"]
EC = df[df['label']=="EC"]
CC = df[df['label']=="CC"]
LGSC = df[df['label']=="LGSC"]
MC = df[df['label']=="MC"]

plt.figure(figsize=(20, 6))
plt.rcParams['font.size'] = 14
colors = ['red', 'lightblue', 'green','magenta', 'yellow']
plt.pie([len(HGSC), len(EC), len(CC), len(LGSC), len(MC)], 
        labels=['HGSC', 'EC', 'CC', 'LGSC', 'MC'], autopct='%1.1f%%', colors=colors)
plt.title('Training Set')
plt.show()
assert sum([len(HGSC), len(EC), len(CC), len(LGSC), len(MC)]) == len(df), "Pie chart proportions must match dataset size"


import seaborn as sns
sns.histplot(data=df, x='label', kde=True, color='blue')
plt.title('Label Distribution with Seaborn')
plt.show()
assert not df['label'].isnull().any(), "Ensure there are no missing values in the label column"



HGSC = df[df['label'] == "HGSC"]
EC = df[df['label'] == "EC"]
CC = df[df['label'] == "CC"]
LGSC = df[df['label'] == "LGSC"]
MC = df[df['label'] == "MC"]

# Correction: Adjusted figure size for better pie chart display
plt.figure(figsize=(10, 10))  # Changed from (20, 6) to (10, 10) for a balanced layout

# Set font size for readability
plt.rcParams['font.size'] = 14

# Create pie chart with appropriate colors and labels
colors = ['red', 'lightblue', 'green', 'magenta', 'yellow']
plt.pie([len(HGSC), len(EC), len(CC), len(LGSC), len(MC)], 
        labels=['HGSC', 'EC', 'CC', 'LGSC', 'MC'], autopct='%1.1f%%', colors=colors)
plt.title('Training Set')  # Set chart title
plt.show()

# Assertion to ensure the pie chart proportions match the dataset size
assert sum([len(HGSC), len(EC), len(CC), len(LGSC), len(MC)]) == len(df), "Pie chart proportions must match dataset size"


import seaborn as sns
import matplotlib.pyplot as plt

# Check for missing values in the 'label' column
if df['label'].isnull().any():
    raise ValueError("Ensure there are no missing values in the 'label' column")

# Create the histogram plot
sns.histplot(x='label', data=df, kde=True, color='green')
plt.title('Label Distribution with Seaborn')
plt.show()



numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Reds_r')
plt.title('Correlation Matrix')
plt.show()
assert correlation_matrix.shape[0] > 0, "Correlation matrix must be generated"



sns.scatterplot(data=df, x='image_id', y='is_tma', hue='label', palette='Blues')
plt.title('Scatter Plot of image_id vs is_tma')
plt.show()
assert 'image_id' in df.columns and 'is_tma' in df.columns, "Ensure 'image_id' and 'is_tma' columns exist in the dataset"


import seaborn as sns
import matplotlib.pyplot as plt

# Make sure the columns 'image_width' and 'image_height' exist
assert 'image_width' in df.columns and 'image_height' in df.columns, "Ensure 'image_width' and 'image_height' columns exist in the dataset"

# Create a scatter plot
sns.scatterplot(data=df, x='image_width', y='image_height', hue='label', palette='Set2')

# Add a title and display the plot
plt.title('Scatter Plot of Image Width vs Image Height')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of 'image_width' vs 'image_height'
sns.scatterplot(x='image_width', y='image_height', hue='label', data=df, palette='Set2')
plt.title('Scatter Plot of Image Width vs Image Height')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the boxplot directly
sns.boxplot(x='label', y='image_height', data=df, palette='cool')
plt.title('Box Plot of Image Height by Label')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")

# Plot the count of 'label'
sns.countplot(x='label', data=df, palette='pastel')
plt.title('Count Plot of Categorical Feature')
plt.show()


