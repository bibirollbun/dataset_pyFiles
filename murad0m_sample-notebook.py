# Random imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import matplotlib.pyplot as plt
import cv2
import random
import os
from collections import Counter


os.environ['KAGGLE_USERNAME']='YOUR_USERNAME'
os.environ['KAGGLE_KEY']='YOUR_KAGGLE_KEY'


!pip install -q -U kaggle


!kaggle competitions download -c dl4cv-coin-classification --force


!unzip dl4cv-coin-classification.zip


train_df = pd.read_csv('kaggle/train.csv')
test_df = pd.read_csv('kaggle/test.csv')

print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)

train_df.head()


num_classes = train_df['Class'].nunique()
print(f'Number of classes: {num_classes}')

class_counts = train_df['Class'].value_counts()
class_counts.head(10)


top_10 = class_counts.head(10)
plt.figure(figsize=(8, 4))
top_10.plot(kind='bar')
plt.title('Top 10 Classes by Image Count')
plt.xlabel('Class Name')
plt.ylabel('Count')
plt.show()


some_id = random.choice(train_df['Id'].values)
image_path = os.path.join('kaggle/train', f'{some_id}.jpg')  # or .png, etc.
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title('Class: ' + train_df.loc[train_df['Id'] == some_id, 'Class'].values[0])
plt.show()




fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

for i, image_id in enumerate([11525, 11522, 11534]):
  image_path = os.path.join('kaggle/train', f'{image_id}.jpg')
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  axes[i].imshow(img)
  axes[i].set_title(f'ID: {image_id}, 50 Cents,Euro,netherlands')
  axes[i].axis('off')

plt.tight_layout()
plt.show()




fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

for i, image_id in enumerate([8780, 11308, 8829]):
  image_path = os.path.join('kaggle/train', f'{image_id}.jpg')
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  axes[i].imshow(img)
  axes[i].set_title(f'ID: {image_id}, Euro,Germany')
  axes[i].axis('off')

plt.tight_layout()
plt.show()




classes = train_df['Class'].unique()
test_ids = test_df['Id'].values

preds = np.random.choice(classes, size=len(test_ids))

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Class': preds
})

submission_df.to_csv('submission.csv', index=False)
submission_df.head(10)


!kaggle competitions submit -c dl4cv-coin-classification -f submission.csv -m "test message"

