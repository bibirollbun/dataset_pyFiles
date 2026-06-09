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


!pip install tensorflow==2.12.0  # PhiÃªn báº£n á»•n Ä‘á»‹nh vá»›i Kaggle


!pip uninstall tensorflow -y
!pip install tensorflow-cpu==2.10.0  # PhiÃªn báº£n á»•n Ä‘á»‹nh


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from xgboost import XGBClassifier
import tensorflow as tf

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Ä�á»�c dá»¯ liá»‡u tá»« Kaggle dataset
data_dir = '/kaggle/input/aptos2019-blindness-detection/train_images'
csv_file = '/kaggle/input/aptos2019-blindness-detection/train.csv'
df = pd.read_csv(csv_file)

# Tiá»�n xá»­ lÃ½ áº£nh vá»›i xá»­ lÃ½ lá»—i
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"KhÃ´ng thá»ƒ Ä‘á»�c áº£nh tá»« {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Lá»—i khi xá»­ lÃ½ áº£nh {image_path}: {str(e)}")
        return None
# Chuáº©n bá»‹ dá»¯ liá»‡u vá»›i thanh tiáº¿n trÃ¬nh
from tqdm import tqdm

images = []
labels = []
failed_images = []

for index, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(data_dir, row['id_code'] + '.png')
    img = preprocess_image(image_path)
    if img is not None:
        images.append(img)
        labels.append(row['diagnosis'])
    else:
        failed_images.append(row['id_code'])

print(f"\nTá»•ng sá»‘ áº£nh khÃ´ng thá»ƒ Ä‘á»�c: {len(failed_images)}")

images = np.array(images)
labels = np.array(labels)


# split data
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# build CNN Feature Extractor
def build_cnn_feature_extractor(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x) 
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)  
    model = Model(inputs=inputs, outputs=x)
    return model
# trainding CNN
cnn_model = build_cnn_feature_extractor()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
print("\nTrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng tá»« CNN...")
train_features = cnn_model.predict(X_train, batch_size=32)
test_features = cnn_model.predict(X_test, batch_size=32)


# trainding XGBoost
print("\nHuáº¥n luyá»‡n XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=7,
    learning_rate=0.05,
    objective='multi:softmax',
    num_class=5,
    tree_method='hist',  
    device='cpu',       
    random_state=42
)

xgb_model.fit(train_features, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# forecast
y_pred = xgb_model.predict(test_features)

# Print report results
accuracy = accuracy_score(y_test, y_pred)
print(f"âœ… Ä�á»™ chÃ­nh xÃ¡c (Accuracy): {accuracy:.4f}")
print("\nğŸ“„ Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Dá»± Ä‘oÃ¡n")
plt.ylabel("Thá»±c táº¿")
plt.title("ğŸ”� Confusion Matrix - XGBoost")
plt.show()



# save model
cnn_model.save('/kaggle/working/cnn_feature_extractor.h5')
xgb_model.save_model('/kaggle/working/xgboost_model.json')

print("\nÄ�Ã£ lÆ°u mÃ´ hÃ¬nh vÃ o thÆ° má»¥c /kaggle/working/")


# Come see samples: https://colab.research.google.com/drive/1XM23__eWWgUbp0F8VxfFqeTwhovUacGS?usp=sharing
# download 2 model vá»�, up lÃªn google drive
# Connect google drive and google colab
# copy path link model and paste 
# run GUI gradio

