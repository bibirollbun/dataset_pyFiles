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
import zipfile
print(os.listdir("/kaggle/input/dogs-vs-cats"))


import zipfile
import os
train_zip_path = "/kaggle/input/dogs-vs-cats/train.zip"
train_unzip_path = "/kaggle/working/train"
os.makedirs(train_unzip_path, exist_ok=True)
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_unzip_path)
print("✅ Train images extraction complete.")


import cv2
import numpy as np
base_dir = "/kaggle/working/train/train"
image_data = []
labels = []
for filename in os.listdir(base_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(base_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        flattened_img = img.flatten()
        image_data.append(flattened_img)
        if "cat" in filename:
            labels.append(0)
        elif "dog" in filename:
            labels.append(1)
image_data = np.array(image_data)
labels = np.array(labels)
image_data = image_data / 255.0
print("✅ Image loading and preprocessing complete.")
print(f"Image data shape: {image_data.shape}")
print(f"Labels shape: {labels.shape}")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)
print("✅ SVM model training complete.")


from sklearn.metrics import accuracy_score
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM model: {accuracy:.4f}")


test1_zip_path = "/kaggle/input/dogs-vs-cats/test1.zip"
test1_unzip_path = "/kaggle/working/test1"
os.makedirs(test1_unzip_path, exist_ok=True)
with zipfile.ZipFile(test1_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test1_unzip_path)
print("✅ Train images extraction complete.")


test_dir = "/kaggle/working/test1/test1"
test_image_data = []
test_image_ids = []
for filename in os.listdir(test_dir):
    if filename.endswith(".jpg"):
        image_id = int(os.path.splitext(filename)[0])
        test_image_ids.append(image_id)
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        flattened_img = img.flatten()
        test_image_data.append(flattened_img)
test_image_data = np.array(test_image_data)
test_image_data = test_image_data / 255.0
print("✅ Test image loading and preprocessing complete.")
print(f"Test image data shape: {test_image_data.shape}")
print(f"Test image IDs shape: {len(test_image_ids)}")


test_predictions = svm_model.predict(test_image_data)
print("✅ Predictions on test set complete.")


import joblib

# Define the path to save the model
model_save_path = "svm_cat_dog_model.pkl"

# Save the trained model
joblib.dump(svm_model, model_save_path)

print(f"✅ SVM model saved successfully to {model_save_path}")


import pandas as pd

# Create a DataFrame with image IDs and predictions
submission_df = pd.DataFrame({'id': test_image_ids, 'label': test_predictions})

# Sort by image ID
submission_df = submission_df.sort_values(by='id')

# Save to submission.csv
submission_df.to_csv('submission.csv', index=False)

print("✅ submission.csv created successfully.")

