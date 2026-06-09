

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




label_df = pd.DataFrame(df['label'].value_counts())
label_df.reset_index(inplace=True)
label_df.columns = ["label","count"]
plt.bar(label_df['label'],label_df['count'], color='skyblue')
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
assert not df['label'].isnull().any(), "Ensure there are no missing values in the label column "


numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
assert correlation_matrix.shape[0] > 0, "Correlation matrix must be generated"


import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
data = {'feature1':[1,2,3,4,5],'feature2':[5,4,3,2,1],'label':['A','B','A','B','A']}
df = pd.DataFrame(data)
assert 'feature1' in df.columns and 'feature2' in df.columns,"ensure'feature1'and 'feature2'columns exist in the dataset"
sns.scatterplot(data=df, x= 'feature1', y= 'feature2', hue = 'label', palette = "summer_r")
plt.title('scatter plot of Feature 1 and feature 2')
plt.show()


sns.scatterplot(data=df, x='feature1', y='feature2', hue='label', palette='coolwarm')
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.show()
assert 'feature1' in df.columns and 'feature2' in df.columns, "Ensure 'feature1' and 'feature2' columns exist in the dataset"


sns.pairplot(data=df, hue='label', palette='husl')
plt.suptitle('Pair Plot of Dataset Features', y=1.02)
plt.show()
assert 'label' in df.columns, "Ensure the 'label' column is included for the pair plot"


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = {'label':['A','B','A','B','C'],'numerical_feature':[1.5,2.3,3.7,4.1,2.9]}
df = pd.DataFrame(data)
assert 'numerical_feature' in df.columns, "Ensure 'numerical_feature' exists in the dataset"
sns.boxplot(data=df, x='label', y='numerical_feature', palette='Set2')
plt.title('Box Plot of Numerical Feature by Label')
plt.show()



sns.boxplot(data=df, x='label', y='numerical_feature', palette='summer_r')
plt.title('Box Plot of Numerical Feature by Label')
plt.show()
assert 'numerical_feature' in df.columns, "Ensure 'numerical_feature' exists in the dataset"


import pandas as pd
data = {'categorical_feature':['A','B','A','C','B','C','A']}
df = pd.DataFrame(data)
import seaborn as sns
import matplotlib.pyplot as plt
assert 'categorical_feature' in df.columns, "Ensure 'categorical_feature' exists in the dataset"
sns.countplot(data=df, x='categorical_feature', palette='coolwarm')
plt.title('Count Plot of Categorical Feature')
plt.show()


sns.countplot(data=df, x='categorical_feature', palette='pastel')
plt.title('Count Plot of Categorical Feature')
plt.show()
assert 'categorical_feature' in df.columns, "Ensure 'categorical_feature' exists in the dataset"

