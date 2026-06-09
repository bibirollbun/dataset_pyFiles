import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import collections
import cv2



train_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

train_df.head()


train_data_dir = "/kaggle/input/ai-vs-human-generated-dataset"
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(train_data_dir, x))
train_df['label'] = train_df['label'].astype(str)



print(train_df['label'].value_counts())



train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode='nearest',
    validation_split=0.1
    
)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='file_name',
    y_col='label',
    class_mode='binary',
    target_size=(128, 128),
    batch_size=32,
    subset='training',
    shuffle=True
)


val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False  )



images, labels = next(train_generator)
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Label: {int(labels[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()


labels_list = []

for i in range(5):  
    _, labels = next(train_generator)
    labels_list.extend(labels)

label_counts = collections.Counter(labels_list)

plt.bar(['Label 0', 'Label 1'], [label_counts[0], label_counts[1]])
plt.title('Class Distribution')
plt.ylabel('Number of Samples')
plt.show()


augmented_images, _ = next(train_generator)

plt.figure(figsize=(12, 4))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(augmented_images[i])
    plt.axis('off')
plt.suptitle('Augmented Samples')
plt.show()



unique, counts = np.unique(train_generator.classes, return_counts=True)
print(dict(zip(unique, counts)))





