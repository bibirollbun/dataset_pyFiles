# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



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


print("Size of the dataset: ", df.shape[0])
assert df.shape[0] > 0, "Dataset size must be greater than 0"


def plot_images(folder_path: str, resize: bool = False):
    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    num_images_to_plot = 4
    selected_images = random.sample(image_files, num_images_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
label_df.columns = ['Ovarian Cancer subtypes', 'count']  
plt.bar(label_df['Ovarian Cancer subtypes'], label_df['count'], color='brown')
plt.xlabel('Ovarian Cancer subtypes')
plt.ylabel('Prevalence')
plt.title('Ovarian Cancer subtypes Distribution')
plt.show()

assert len(label_df) > 0, "Label distribution must be displayed correctly"


label_df = pd.DataFrame(df['label'].value_counts())
label_df.reset_index(inplace=True)
label_df.columns = ['Ovarian Cancer subtypes', 'count']

label_df['color'] = pd.cut(label_df['count'], bins=len(['#FF7F7F', '#FF3030', '#DC143C', '#8B0000']), labels=['#FF7F7F', '#FF3030', '#DC143C', '#8B0000'])

plt.bar(label_df['Ovarian Cancer subtypes'], label_df['count'], color=label_df['color'])
plt.xlabel('Ovarian Cancer subtypes')
plt.ylabel('Prevalence')
plt.title('Ovarian Cancer subtypes Distribution')
plt.show()

assert len(label_df) > 0, "Label distribution must be displayed correctly"


import seaborn as sns
sns.histplot(data=df, x='label', kde=True, color='#8B0000')
plt.title('Subtype Distribution with Seaborn')
plt.show()
assert not df['label'].isnull().any(), "Ensure there are no missing values in the label column"


numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu')
plt.title('Correlation Matrix')
plt.show()
assert correlation_matrix.shape[0] > 0, "Correlation matrix must be generated"


sns.scatterplot(data=df, x='image_id', y='is_tma', hue='label', palette='RdYlBu')
plt.title('Scatter Plot of image_id vs is_tma')
plt.show()
assert 'image_id' in df.columns and 'is_tma' in df.columns, "Ensure 'image_id' and 'is_tma' columns exist in the dataset"


sns.pairplot(data=df, hue='label', palette='hsv')
plt.suptitle('Pair Plot of Dataset Features', y=1.02)
plt.show()
assert 'label' in df.columns, "Ensure the 'label' column is included for the pair plot"


sns.boxplot(data=df, x='label', y='image_height', palette='rainbow')
plt.title('Box Plot of Image Height by Subtype')
plt.show()
assert 'image_height' in df.columns, "Ensure 'image_height' exists in the dataset"


sns.countplot(data=df, x='label', palette='twilight')
plt.title('Count Plot of label')
plt.show()
assert 'label' in df.columns, "Ensure 'label' exists in the dataset"

