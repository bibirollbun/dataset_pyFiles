

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



df.shape



df.info()


df.isnull().sum()



# Verify dataset size
print("Size of the dataset: ", df.shape[0])
assert df.shape[0] > 0, "Dataset size must be greater than 0"


Image.MAX_IMAGE_PIXELS = None #Increase the limit for image pixels to avoid DecompressionBombError
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
folder_path = '/kaggle/input/UBC-OCEAN/train_images'  # Replace with your actual path
plot_images(folder_path, resize=True)



label_df = pd.DataFrame(df['label'].value_counts())
label_df.reset_index(inplace=True)
label_df.columns = ['label', 'count']  # Rename the columns for clarity

plt.bar(label_df['label'], label_df['count'], color='skyblue')  # Use the renamed columns
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


# Filter rows by label to calculate proportions
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
sns.histplot(data=df, x='label', kde=True, color='blue')
plt.title('Label Distribution with Seaborn')
plt.show()
assert not df['label'].isnull().any(), "Ensure there are no missing values in the label column"


#corrected version making sure there are noinf error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Replace inf and -inf values in the 'label' column with NaN
df['label'] = df['label'].replace([np.inf, -np.inf], np.nan)  # Correction: Handle infinite values to avoid plotting issues

# Drop rows with NaN values in the 'label' column
df = df.dropna(subset=['label'])  # Correction: Ensure no NaN values remain in the 'label' column

# Use sns.countplot to plot the label distribution
sns.countplot(data=df, x='label', palette='Blues')  # Visualization remains the same

# Set the title and axis labels for clarity
plt.title('Label Distribution with Seaborn')
plt.xlabel('Labels')  # Added label for the x-axis
plt.ylabel('Count')   # Added label for the y-axis

# Display the plot
plt.show()

# Assertion to confirm the 'label' column is clean
assert not df['label'].isnull().any(), "Ensure there are no missing or inf values in the label column"  # Final check to guarantee data integrity



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")

# Calculate the correlation matrix for numeric columns
numeric_df = train_df.select_dtypes(include=[np.number])

# Plot the correlation matrix
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Training Data')
plt.show()

# Ensure the correlation matrix is valid
assert numeric_df.corr().shape[0] > 0, "Correlation matrix must be generated"



sns.scatterplot(data=df="/kaggle/input/UBC-OCEAN/test.csv", x='feature1', y='feature2', hue='label', palette='Set2')
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.show()
assert 'feature1' in df.columns and 'feature2' in df.columns, "Ensure 'feature1' and 'feature2' columns exist in the dataset"


 import pandas as pd

# Load the dataset
df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")

# Print the column names to confirm what is available
print(df.columns)



import seaborn as sns
import matplotlib.pyplot as plt

# Make sure the columns 'image_width' and 'image_height' exist
assert 'image_width' in df.columns and 'image_height' in df.columns, "Ensure 'image_width' and 'image_height' columns exist in the dataset"

# Create a scatter plot
sns.scatterplot(data=df, x='image_width', y='image_height', hue='label', palette='Set2')

# Add a title and display the plot
plt.title('Scatter Plot of Image Width vs Image Height')
plt.show()



sns.pairplot(data=df, hue='label', palette='husl')
plt.suptitle('Pair Plot of Dataset Features', y=1.02)
plt.show()
assert 'label' in df.columns, "Ensure the 'label' column is included for the pair plot"


import seaborn as sns
import matplotlib.pyplot as plt

# Create the pair plot with 'label' as hue
sns.pairplot(data=df, hue='label', palette='husl')

# Set title for the plot
plt.suptitle('Pair Plot of Dataset Features', y=1.02)

# Show the plot
plt.show()



sns.boxplot(data=df, x='label', y='numerical_feature', palette='cool')
plt.title('Box Plot of Numerical Feature by Label')
plt.show()
assert 'numerical_feature' in df.columns, "Ensure 'numerical_feature' exists in the dataset"


import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'image_width' exists in the dataframe
assert 'image_width' in df.columns, "Ensure 'image_width' exists in the dataset"

# Create a boxplot using 'image_width' as the numerical feature
sns.boxplot(data=df, x='label', y='image_width', palette='cool')

# Set title for the plot
plt.title('Box Plot of Image Width by Label')

# Show the plot
plt.show()




df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")  # Replace with your actual path to the dataset

sns.countplot(data=df, x='categorical_feature', palette='pastel')
plt.title('Count Plot of Categorical Feature')
plt.show()
assert 'categorical_feature' in df.columns, "Ensure 'categorical_feature' exists in the dataset"


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")  # Adjust path as necessary

# Ensure 'label' exists in the dataframe
assert 'label' in df.columns, "Ensure 'label' exists in the dataset"

# Create a count plot using 'label' as the categorical feature
sns.countplot(data=df, x='label', palette='pastel')

# Set the title for the plot
plt.title('Count Plot of Categorical Feature')

# Show the plot
plt.show()


