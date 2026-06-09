





# Import libraries
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt


train0=pd.read_csv('/kaggle/input/happy-whale-and-dolphin/train.csv')
print(train0.columns.tolist())
print(len(train0))


id_counts = train0['individual_id'].value_counts()
valid_ids = id_counts[id_counts >= 100].index
train = train0[train0['individual_id'].isin(valid_ids)]
#train = train[train['individual_id']!='new_whale'].reset_index(drop=True)
print(len(train))


file2id=train.set_index("image")["individual_id"].to_dict()
unique_ids=sorted(train['individual_id'].unique().tolist())
print(len(unique_ids))
print(unique_ids[0:5])


display(train)


dir0='/kaggle/input/happywhale-cropped-removebackground-v1/removedBackground_train_images'
paths=[]
ids=[]
for dirname, _, filenames in os.walk(dir0):
    for filename in filenames:
        path=os.path.join(dirname, filename)
        Id=(train0[train0['image']==filename]).iloc[0,2]
        if Id in unique_ids:
            paths+=[path]
            ids+=[Id]
print(paths[0:3])
print(ids[0:3])


# 1. Feature Extraction (HOG)
def extract_hog_features(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        image = cv2.resize(image, (128,128))  # Standardize size
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            visualize=False  # Explicitly set to False (the default)
        )
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


image_path=paths[0]
print(image_path)
extract_hog_features(image_path)


# 2. Load dataset from folders
MAX_FEATURES = 8100  # Example for 128x128 image with default HOG params
features, labels = [], []
for Id,path in zip(ids,paths):
    hog_features = extract_hog_features(path)  #
    if hog_features is not None:  # Skip failed images
        features.append(hog_features[:MAX_FEATURES])
        labels.append(Id)


# 3. Load data and split
class_names=sorted(os.listdir(dir0))
X=np.array(features)
y=np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


img=plt.imread(paths[0])
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.colorbar()
plt.axis('off')
plt.show()


print(X.shape)
vec = X[0]
img_like = vec.reshape(90,90)
plt.imshow(img_like, cmap='hot')
plt.title("HOG Vector Heatmap")
plt.colorbar()
plt.axis('off')
plt.show()


# 4. Train Random Forest
clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
clf.fit(X_train, y_train)


# 5. Evaluate
y_true = y_test
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred,target_names=unique_ids,digits=4))




