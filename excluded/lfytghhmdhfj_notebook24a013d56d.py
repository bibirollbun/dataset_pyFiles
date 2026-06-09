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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from PIL import Image



# Load the training CSV
df = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
label_map = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}
df['label_name'] = df['label'].map(label_map)
df['image_path'] = '/kaggle/input/cassava-leaf-disease-classification/train_images/' + df['image_id']
df.head()



import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import cv2
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D,Dropout,BatchNormalization,Reshape
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D,Dropout,BatchNormalization,Reshape
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import math
from tensorflow.keras.models import load_model


plt.figure(figsize=(25, 8))
ax = sns.countplot(
    x=df["label_name"],
    palette="viridis",
    order=df['label_name'].value_counts().index
)
ax.set_title("Distribution of Cassava Leaf Disease Labels", fontsize=22)
ax.set_xlabel("Disease Class", fontsize=18)
ax.set_ylabel("Image Count", fontsize=18)
ax.tick_params(labelsize=14)

# Add value labels on top of bars
for p in ax.containers:
    ax.bar_label(p, fontsize=14, color='black', padding=5)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()



import cv2
import numpy as np
from tqdm import tqdm  # Optional: shows a progress bar

x = []
for image_path in tqdm(df['image_path']):  # Using the precomputed full path
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        x.append(img)
    else:
        print(f"Warning: Could not read image at path: {image_path}")

x = np.array(x)
print("Shape of image data:", x.shape)



y=df[["label"]]



x.shape,y.shape



import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class ManualSMOTE:
    def __init__(self, sampling_strategy='auto', random_state=None, k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        class_counts = Counter(y)
        max_count = max(class_counts.values())

        new_X = []
        new_y = []

        for cls in class_counts:
            X_cls = X[y == cls]
            n_samples = len(X_cls)
            n_generate = max_count - n_samples

            if n_generate <= 0:
                continue

            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_samples))
            nn.fit(X_cls)
            neighbors = nn.kneighbors(X_cls, return_distance=False)

            for _ in range(n_generate):
                i = np.random.randint(0, n_samples)
                neighbor_idx = np.random.choice(neighbors[i][1:])  # skip self
                diff = X_cls[neighbor_idx] - X_cls[i]
                gap = np.random.rand()
                new_sample = X_cls[i] + gap * diff
                new_X.append(new_sample)
                new_y.append(cls)

        X_resampled = np.vstack([X, np.array(new_X)])
        y_resampled = np.hstack([y, np.array(new_y)])

        return X_resampled, y_resampled



x_flat = x.reshape(x.shape[0], -1)
y_flat = y['label'].values

smote = ManualSMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_flat, y_flat)
x_resampled_images = x_resampled.reshape(-1, 32, 32, 3)



x_resampled_images.shape,y_resampled.shape


plt.figure(figsize=(25,8))
y_resampled_series = pd.Series(y_resampled)
ax=sns.countplot(x=y_resampled_series,palette="viridis")
for p in ax.containers:
    ax.bar_label(p, fontsize=12, color='black', padding=5);


from sklearn.model_selection import train_test_split

# Split into 80% train and 20% validation
x_train, x_val, y_train, y_val = train_test_split(
    x_resampled_images, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print("Train samples:", len(x_train))
print("Validation samples:", len(x_val))



model=Sequential()
model.add(Input(shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Callbacks
checkpoint = ModelCheckpoint(
    "best_model.keras",           # ✅ Updated extension
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-5,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Fit the model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=35,
    batch_size=36,
    verbose=1,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)



import cv2
import os
import numpy as np
import pandas as pd

# Load sample_submission.csv to get test image IDs
sample_submission = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv")

# Path to test images
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

# Preprocess all test images
test_images = []
test_image_ids = []

for image_id in sample_submission['image_id']:
    image_path = os.path.join(test_dir, image_id)
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        test_images.append(img)
        test_image_ids.append(image_id)
    else:
        print(f"Could not load image: {image_id}")

x_test_final = np.array(test_images)



import os
import cv2
import numpy as np
import pandas as pd

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv")

x_kaggle_test = []

for image_id in test_df['image_id']:
    path = os.path.join(test_dir, image_id)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        x_kaggle_test.append(img)
    else:
        print(f"Could not read image: {image_id}")

x_kaggle_test = np.array(x_kaggle_test)



y_kaggle_pred_probs = model.predict(x_kaggle_test)
y_kaggle_pred = np.argmax(y_kaggle_pred_probs, axis=1)



submission = test_df.copy()
submission['label'] = y_kaggle_pred
submission.to_csv('submission.csv', index=False)





