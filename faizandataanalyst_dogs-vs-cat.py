import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import zipfile



zip_path = "/kaggle/input/dogs-vs-cats/train.zip"
extract_path = "/kaggle/working/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

train_dir = "/kaggle/working/train"

print(f" Extracted training images to: {extract_path}")



base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)



def extract_features(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()



cat_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.startswith("cat")]
dog_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.startswith("dog")]

cat_images = cat_images[:10000]
dog_images = dog_images[:10000]

image_paths = cat_images + dog_images
labels = [0]*len(cat_images) + [1]*len(dog_images)



X = []
for path in tqdm(image_paths, desc="ğŸ”� Extracting features"):
    feat = extract_features(path)
    X.append(feat)

X = np.array(X)
y = np.array(labels)
print(" Features shape:", X.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"ğŸ”€ Split: {X_train.shape[0]} train | {X_test.shape[0]} test")



clf = LinearSVC(C=1.0, max_iter=2000)
clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy*100:.2f}%")




print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Example for TensorFlow / Keras
model.save('imageclassifier.h5')
# OR for PyTorch
torch.save(model.state_dict(), 'imageclassifier.pth')


