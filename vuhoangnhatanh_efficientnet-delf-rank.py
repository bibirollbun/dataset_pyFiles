!nvidia-smi


import os
import cv2
import shutil
import numpy as np
import pandas as pd
from scipy import spatial
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Directories and file paths
TRAIN_DIR = '/kaggle/input/landmark-recognition-2021/train'
TRAIN_CSV = '/kaggle/input/landmark-recognition-2021/train.csv'
train_df = pd.read_csv(TRAIN_CSV)

# Construct full image paths
TRAIN_PATHS = [os.path.join(TRAIN_DIR, f'{img[0]}/{img[1]}/{img[2]}/{img}.jpg') for img in train_df['id']]
train_df['path'] = TRAIN_PATHS

# Count occurrences of each landmark_id
train_df_grouped = train_df['landmark_id'].value_counts().reset_index()
train_df_grouped.columns = ['landmark_id', 'count']

# Select top 100 most frequent landmarks
selected_landmarks = train_df_grouped.head(100)

# Subset the main dataframe
train_df_sub = train_df[train_df['landmark_id'].isin(selected_landmarks['landmark_id'])]

# Map landmark_id to new_id (0 to 99)
landmark_id_map = {old_id: new_id for new_id, old_id in enumerate(selected_landmarks['landmark_id'])}
train_df_sub['new_id'] = train_df_sub['landmark_id'].map(landmark_id_map)

# Number of classes
NUM_CLASSES = train_df_sub['new_id'].nunique()
print(f"Unique classes found: {NUM_CLASSES}")

# Display the final subset
train_df_sub.head()



train_df_sub


# Training and validation splits
# 90/10 stratified split for training and validation
X_train, X_val, y_train, y_val = train_test_split(train_df_sub[['id', 'path']], train_df_sub['new_id'],
                                                  train_size = 0.9,
                                                  random_state = 123,
                                                  shuffle = True,
                                                  stratify = train_df_sub['new_id'])

# Held-out test set for inference
# Further 95/5 split -> 5% of original training set left for test set
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                   train_size = 0.90,
                                                   random_state = 123,
                                                   shuffle = True,
                                                   stratify = y_train)

assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == train_df_sub.shape[0]

print(f"Training data shape: {X_train.shape}")
print(f"Training label shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation label shape: {y_val.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test label shape: {y_test.shape}")


print(f"Unique classes on y_train: {y_train.nunique()}")
print(f"Unique classes on y_val: {y_val.nunique()}")
print(f"Unique classes on y_test: {y_test.nunique()}")


# Classes distribution on training, validation and test sets
plt.figure(figsize = (10, 3))
ax = sns.histplot(y_train, bins=75, kde = True)
ax.set_title('Distribution of Landmarks on training set')
plt.tight_layout()

plt.figure(figsize = (10, 3))
ax = sns.histplot(y_val, bins=75, kde = True)
ax.set_title('Distribution of Landmarks on validation set')
plt.tight_layout()

plt.figure(figsize = (10, 3))
ax = sns.histplot(y_test, bins=75, kde = True)
ax.set_title('Distribution of Landmarks on test set')
plt.tight_layout()
plt.show()


# Creating image directories for classes subset
NEW_BASE_DIR = "/kaggle/working"

# Training set directory
for file, path, landmark in tqdm(zip(X_train['id'], X_train['path'], y_train)):
    dir = f"{NEW_BASE_DIR}/train_sub/{str(landmark)}"
    os.makedirs(dir, exist_ok = True)
    fname = f"{file}.jpg"
    shutil.copyfile(src = path, dst = f"{dir}/{fname}")

# Validation set directory    
for file, path, landmark in tqdm(zip(X_val['id'], X_val['path'], y_val)):
    dir = f"{NEW_BASE_DIR}/val_sub/{str(landmark)}"
    os.makedirs(dir, exist_ok = True)
    fname = f"{file}.jpg"
    shutil.copyfile(src = path, dst = f"{dir}/{fname}")

# Training set directory
for file, path, landmark in tqdm(zip(X_test['id'], X_test['path'], y_test)):
    dir = f"{NEW_BASE_DIR}/test_sub/{str(landmark)}"
    os.makedirs(dir, exist_ok = True)
    fname = f"{file}.jpg"
    shutil.copyfile(src = path, dst = f"{dir}/{fname}")


!ls


!cd train_sub && ls


torch.cuda.memory_summary(device=None, abbreviated=False)


# Creating tensorflow tf.data.Dataset
from tensorflow.keras.utils import image_dataset_from_directory

IMG_SIZE = 224
BATCH_SIZE = 16

print("Building training dataset...")
# Training tf.data.Dataset
train_ds = image_dataset_from_directory(f"{NEW_BASE_DIR}/train_sub",
                                        label_mode = 'int',
                                        shuffle = True,
                                        image_size = (IMG_SIZE, IMG_SIZE),
                                        batch_size = BATCH_SIZE)

print("Building validation dataset...")
# Validation tf.data.Dataset
val_ds = image_dataset_from_directory(f"{NEW_BASE_DIR}/val_sub",
                                        label_mode = 'int',
                                        shuffle = True,
                                        image_size = (IMG_SIZE, IMG_SIZE),
                                        batch_size = BATCH_SIZE)

print("Building test dataset...")
# Test tf.data.Dataset
test_ds = image_dataset_from_directory(f"{NEW_BASE_DIR}/test_sub",
                                        label_mode = 'int',
                                        shuffle = True,
                                        image_size = (IMG_SIZE, IMG_SIZE),
                                        batch_size = BATCH_SIZE)


# Visualizing a random batch from training dataset
for data_batch, labels_batch in train_ds.take(1):
    ncols = 4
    nrows = int(data_batch.shape[0]/ncols)
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(10, 11),
                           sharex = True, sharey = True)
    img_counter = 0
    for image, label in zip(data_batch, labels_batch):
        axi = ax.flat[img_counter]
        axi.imshow(image/255.)
        label = label.numpy()
#         axi.set_title(np.where(label == 1)[0])
        axi.set_title(label)
        img_counter += 1
plt.show()


import albumentations as A
from albumentations.pytorch import ToTensorV2

# Defining a data augmentation stage
train_transform = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ImageCompression(quality_lower=99, quality_upper=100),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.CLAHE(p=0.1),
    A.GaussianBlur(p=0.1),
    A.Normalize(),
    ToTensorV2()
])

def __getitem__(self, idx):
    image = cv2.imread(self.image_paths[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = self.transform(image=image)
    return {"image": augmented["image"], "path": self.image_paths[idx]}


# Auxiliar functions
# Load image
def get_image(path, resize = False, reshape = False, target_size = None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, dsize = (target_size, target_size))
    if reshape:
        img = tf.reshape(img, [1, target_size, target_size, 3])
    return img

# Get landmark samples
def get_landmark(landmark_id, samples = 16):
    nrows = samples // 4
    random_imgs = np.random.choice(train_df_sub[train_df_sub['new_id'] == landmark_id].index, samples, replace = False)
    plt.figure(figsize = (12, 10))
    for i, img in enumerate(train_df_sub.loc[random_imgs, :].values):
        ax = plt.subplot(nrows, 4, i + 1)
        plt.imshow(get_image(img[2]))
        plt.title(f"{img[0]}")
        plt.suptitle(f"Samples of landmark {landmark_id}", fontsize = 14, y = 0.94, weight = "bold")
        plt.axis("off")

# Get image embeddings
def get_embeddings(model, image_paths, input_size, as_df = True):
    embeddings = {}
    embeddings['images_paths'] = []
    embeddings['embedded_images'] = []
    
    target_dir = os.path.split(os.path.split(image_paths[0])[0])[0]
    
    print(f"Retrieving embeddings for {target_dir} with {model.name}...")
    for image_path in tqdm(image_paths):
        embeddings['images_paths'].append(image_path)
        embedded_image = model.predict(get_image(image_path,
                                                 resize = True,
                                                 reshape = True,
                                                 target_size = input_size))
        embeddings['embedded_images'].append(embedded_image)
    
    if as_df:
        embeddings = pd.DataFrame(embeddings)
    
    return embeddings

# Get similarities between query key pair
def get_similarities(query, key):
    '''
    Get cosine similarity matrix between query and key pairs
    Arguments:
    query, key: embedded images
    '''
    query_array = np.stack(query.tolist()).reshape(query.shape[0],
                                                   query[0].shape[1])
    key_array = np.stack(key.tolist()).reshape(key.shape[0],
                                               key[0].shape[1])
    
    # Initializing similarity matrix
    similarity = np.zeros((query_array.shape[0], key_array.shape[0]))
    
    # Getting pairwise similarities
    print(f"Getting pairwise {query_array.shape[0]} query: {key_array.shape[0]} key similarities...")
    for query_index in tqdm(range(query_array.shape[0])):
        similarity[query_index] = 1 - spatial.distance.cdist(query_array[np.newaxis, query_index, :],
                                                             key_array,
                                                             'cosine')[0]
    return similarity

# Plot top ranked images
def plot_similar(similar_imgs, img_paths):
    '''
    Plot top N similar samples from similarity index
    '''
    plt.figure(figsize = (18, 6))
    nrows = similar_imgs.shape[0]//5
    for i, img in enumerate(similar_imgs):
        ax = plt.subplot(nrows, 5, i + 1)
        plt.imshow(get_image(img_paths[img]))
        plt.title(f"Landmark id: {os.path.split(os.path.split(img_paths[img])[0])[1]}")
        plt.axis("off")


# Aggregate query and top similar plots
def query_top(image_index, top_n = 5, figsize = (6, 6), reranked = None):
    '''
    Plot top N similar samples against queried image
    If reranked, provide reranked dataframe with ['top_similar'] index reordered by reranked confidence
    '''
    image_id = os.path.split(val_embeddings['images_paths'][image_index])[1]
    query_landmark_id = os.path.split(os.path.split(val_embeddings['images_paths'][image_index])[0])[1]
    
    if type(reranked) == pd.core.frame.DataFrame:
        similar_n = reranked['top_similar'][:top_n]
    else:
        similar_n = np.argsort(val_train_similarity[image_index])[::-1][:top_n]
                
    print(f"Queried image: {image_id}")
    plt.figure(figsize = figsize)
    plt.imshow(get_image(val_embeddings['images_paths'][image_index]))
    plt.title(f"Landmark id: {query_landmark_id}")
    plt.axis("off")
    plot_similar(similar_n, train_embeddings['images_paths'])


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
import tensorflow as tf

IMG_SIZE = 224  

# Load EfficientNetB0 with custom top layer
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', pooling='avg')
x = tf.keras.layers.Dense(512, activation='relu', name='embedding_512')(base_model.output)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)


# Embedding models
embedding_layer = 'embedding_512'
embedding_model = tf.keras.Model(inputs = model.input,
                                 outputs = model.get_layer(embedding_layer).output,
                                 name = "EfficientNetB0_embed512")


# Retrieving embeddings
train_img_paths = train_ds.file_paths
val_img_paths = val_ds.file_paths

train_embeddings = get_embeddings(model = embedding_model,
                                 image_paths = train_img_paths,
                                 input_size = IMG_SIZE)

val_embeddings = get_embeddings(model = embedding_model,
                                 image_paths = val_img_paths,
                                 input_size = IMG_SIZE)


train_embeddings.head()


val_embeddings.head()


val_train_similarity = get_similarities(val_embeddings['embedded_images'],
                                        train_embeddings['embedded_images'])
val_train_similarity.shape


# Calculating confidence score per submission
def confidence_top(query = None, key = None, similarity = None, query_image_index = None, top = 5):
    '''
    Arguments:
    query_image_index = index of query image on similarity matrix query axis
    Return confidence scores for top N predictions
    '''
    query_paths = query['images_paths']
    key_paths = key['images_paths']
    
    similar_n = np.argsort(similarity[query_image_index])[::-1][:top]
    
    confidence_df = {}    
    confidence_df['top_similar'] = []
    for similar in similar_n:
        confidence_df['top_similar'].append(similar)

    confidence_df['image_paths'] = []
    for similar in similar_n:
        similar_image_path = key_paths[similar]
        confidence_df['image_paths'].append(similar_image_path)    
        
    confidence_df['prediction'] = []
    for similar in similar_n:
        similar_image_path = key_paths[similar]
        y = int(os.path.split(os.path.split(similar_image_path)[0])[1])
        confidence_df['prediction'].append(y)  
    
    confidence_df['cos_similarity'] = []
    for similar in similar_n:
        confidence_df['cos_similarity'].append(similarity[query_image_index][similar]) 
    
    return pd.DataFrame(confidence_df)


query_image_index = 0

query_top(query_image_index)


top_n = 100


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


query_image_index = 4

query_top(query_image_index)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


query_image_index = 65

query_top(query_image_index)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


query_image_index = 13

query_top(query_image_index)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


query_image_index = 889
top_n = 10

query_top(query_image_index, top_n)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


DELF_IMG_SIZE = 600


query = val_embeddings              # dictionary with 'embedded_images' and 'images_paths'
key = train_embeddings              # same structure as query
similarity = val_train_similarity  # cosine similarity matrix



confidence_df = confidence_top(query, key, similarity, query_image_index, top=75)

similar_n = confidence_df['top_similar'].values


image_1 = get_image(val_embeddings['images_paths'][889],
                    resize = True,
                    target_size = DELF_IMG_SIZE)

plt.figure(figsize = (6, 6))
plt.imshow(image_1)
plt.axis("off")
plt.show()


similar_n = confidence_df['top_similar'].values


image_2 = get_image(train_embeddings['images_paths'][similar_n[5]],
                     resize = True,
                     target_size = DELF_IMG_SIZE)

plt.figure(figsize = (6, 6))
plt.imshow(image_2)
plt.axis("off")
plt.show()


from absl import logging
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import tensorflow_hub as hub
from six.moves.urllib.request import urlopen


delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']


# DELF module
def run_delf(image):
    '''
    Apply DELF module to the input image
    Arguments:
    image: np.array resized image
    '''
    float_image = tf.image.convert_image_dtype(image, tf.float32)

    return delf(
      image = float_image,
      score_threshold = tf.constant(100.0),
      image_scales = tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num = tf.constant(1000))

def match_images(image1, image2, result1, result2, verbose = True):
    distance_threshold = 0.8

    # Read features.
    num_features_1 = result1['locations'].shape[0]
    num_features_2 = result2['locations'].shape[0]
    
    if verbose:
        print("Loaded image 1's %d features" % num_features_1)
        print("Loaded image 2's %d features" % num_features_2)

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(result1['descriptors'])
    _, indices = d1_tree.query(
      result2['descriptors'],
      distance_upper_bound=distance_threshold)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
      result2['locations'][i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
      result1['locations'][indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC.
    _, inliers = ransac(
      (locations_1_to_use, locations_2_to_use),
      AffineTransform,
      min_samples=3,
      residual_threshold=20,
      max_trials=1000)
    
    if verbose:
        print('Found %d inliers' % sum(inliers))

    # Visualize correspondences.
    _, ax = plt.subplots(figsize = (9, 9))
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(
      ax,
      image1,
      image2,
      locations_1_to_use,
      locations_2_to_use,
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b')
    ax.axis('off')
    ax.set_title(f'DELF correspondences: Found {sum(inliers)} inliers')


delf_result1 = run_delf(image_1)
delf_result2 = run_delf(image_2)


match_images(image_1, image_2, delf_result1, delf_result2)


for image_index in similar_n[:6]:
    key_image = get_image(train_embeddings['images_paths'][image_index],
                          resize = True,
                          target_size = DELF_IMG_SIZE)
    try:
        delf_key_image_result = run_delf(key_image)
        match_images(image_1, key_image, delf_result1, delf_key_image_result, verbose = False)
    except:
        print("No inliers found")


def delf_rerank(query = None, key = None, query_image_index = None, confidence_df = None, re_sort = True):
    distance_threshold = 0.8
    query_paths = query['images_paths']
    key_paths = key['images_paths']
    
    query_image = get_image(query_paths[query_image_index],
                            resize = True,
                            target_size = DELF_IMG_SIZE)
    
    delf_result_query = run_delf(query_image)
    
    # Read query features
    num_features_query = delf_result_query['locations'].shape[0]
    
    inliers_list = []
    print(f"Retrieving local features for top {len(confidence_df['image_paths'])} key images...")
    for image_path in tqdm(confidence_df['image_paths']):
        key_image = get_image(image_path,
                          resize = True,
                          target_size = DELF_IMG_SIZE)
        
        delf_result_key = run_delf(key_image)
    
        # Read key features
        num_features_key = delf_result_key['locations'].shape[0]

        # Find nearest-neighbor matches using a KD tree.
        d1_tree = cKDTree(delf_result_query['descriptors'])
        _, indices = d1_tree.query(
          delf_result_key['descriptors'],
          distance_upper_bound=distance_threshold)

        # Select feature locations for putative matches.
        locations_k_to_use = np.array([
          delf_result_key['locations'][i,]
          for i in range(num_features_key)
          if indices[i] != num_features_query
        ])
        locations_q_to_use = np.array([
          delf_result_query['locations'][indices[i],]
          for i in range(num_features_key)
          if indices[i] != num_features_query
        ])

        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac(
              (locations_q_to_use, locations_k_to_use),
              AffineTransform,
              min_samples=3,
              residual_threshold=20,
              max_trials=1000)
        except:
            inliers = [0]
        
        # Handling 0 inliers
        try:
            total_inliers = sum(inliers)
            inliers_list.append(total_inliers)
        except:
            inliers_list.append(1) # Appending inlier = 1 to avoid null confidence
    
    confidence_df['inliers'] = inliers_list
    
    original_confidence = confidence_df['inliers']
    reranked_confidence = np.sqrt(original_confidence) * confidence_df['cos_similarity']
    confidence_df['reranked_conf'] = reranked_confidence
    
    if re_sort:
        confidence_df.sort_values('reranked_conf', ascending = False, inplace = True)
    
    return confidence_df


reranked_df = delf_rerank(query = val_embeddings,
                          key = train_embeddings,
                          query_image_index = query_image_index,
                          confidence_df = confidence_df,
                          re_sort = True)
reranked_df


query_image_index = 889
top_n = 10

query_top(query_image_index, top_n, reranked = reranked_df)


def delf_rerank(query = None, key = None, query_image_index = None, confidence_df = None, re_sort = True):
    distance_threshold = 0.8
    query_paths = query['images_paths']
    key_paths = key['images_paths']
    
    query_image = get_image(query_paths[query_image_index],
                            resize = True,
                            target_size = DELF_IMG_SIZE)
    
    delf_result_query = run_delf(query_image)
    
    # Read query features
    num_features_query = delf_result_query['locations'].shape[0]
    
    inliers_list = []
    print(f"Retrieving local features for top {len(confidence_df['image_paths'])} key images...")
    for image_path in tqdm(confidence_df['image_paths']):
        key_image = get_image(image_path,
                          resize = True,
                          target_size = DELF_IMG_SIZE)
        
        delf_result_key = run_delf(key_image)
    
        # Read key features
        num_features_key = delf_result_key['locations'].shape[0]

        # Find nearest-neighbor matches using a KD tree.
        d1_tree = cKDTree(delf_result_query['descriptors'])
        _, indices = d1_tree.query(
          delf_result_key['descriptors'],
          distance_upper_bound=distance_threshold)

        # Select feature locations for putative matches.
        locations_k_to_use = np.array([
          delf_result_key['locations'][i,]
          for i in range(num_features_key)
          if indices[i] != num_features_query
        ])
        locations_q_to_use = np.array([
          delf_result_query['locations'][indices[i],]
          for i in range(num_features_key)
          if indices[i] != num_features_query
        ])

        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac(
              (locations_q_to_use, locations_k_to_use),
              AffineTransform,
              min_samples=3,
              residual_threshold=20,
              max_trials=1000)
        except:
            inliers = [0]
        
        # Handling 0 inliers
        try:
            total_inliers = sum(inliers)
            inliers_list.append(total_inliers)
        except:
            inliers_list.append(1) # Appending inlier = 1 to avoid null confidence
    
    confidence_df['inliers'] = inliers_list
    
    original_confidence = confidence_df['inliers']
    reranked_confidence = np.sqrt(original_confidence) * confidence_df['cos_similarity']
    confidence_df['reranked_conf'] = reranked_confidence
    
    if re_sort:
        confidence_df.sort_values('reranked_conf', ascending = False, inplace = True)
    
    return confidence_df


query_image_index = 11
top_n = 10

query_top(query_image_index, top_n)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


reranked_df = delf_rerank(query = val_embeddings,
                          key = train_embeddings,
                          query_image_index = query_image_index,
                          confidence_df = confidence_df,
                          re_sort = True)
reranked_df


query_image_index = 11
top_n = 10

query_top(query_image_index, top_n, reranked = reranked_df)


query_image_index = 395
top_n = 10

query_top(query_image_index, top_n)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


reranked_df = delf_rerank(query = val_embeddings,
                          key = train_embeddings,
                          query_image_index = query_image_index,
                          confidence_df = confidence_df,
                          re_sort = True)
reranked_df


query_image_index = 395
top_n = 10

query_top(query_image_index, top_n, reranked = reranked_df)


query_image_index = 40
top_n = 10

query_top(query_image_index, top_n)


confidence_df = confidence_top(query = val_embeddings,
                               key = train_embeddings,
                               similarity = val_train_similarity,
                               query_image_index = query_image_index,
                               top = top_n)

confidence_df


reranked_df = delf_rerank(query = val_embeddings,
                          key = train_embeddings,
                          query_image_index = query_image_index,
                          confidence_df = confidence_df,
                          re_sort = True)
reranked_df


query_image_index = 40
top_n = 10

query_top(query_image_index, top_n, reranked = reranked_df)


query_image_index = 822
top_n = 10

query_top(query_image_index, top_n)


get_landmark(36)


def recognize_and_visualize(query_image_path, 
                            train_embeddings, 
                            model, 
                            top_n=5, 
                            use_rerank=True):
    # Step 1: Get query embedding
    query_embedding = get_embeddings(model=model,
                                     image_paths=[query_image_path],
                                     input_size=IMG_SIZE)

    # Step 2: Compute cosine similarity
    sim_matrix = get_similarities(query_embedding['embedded_images'],
                                   train_embeddings['embedded_images'])

    # Step 3: Get top-N predictions
    confidence_df = confidence_top(query=query_embedding,
                                   key=train_embeddings,
                                   similarity=sim_matrix,
                                   query_image_index=0,
                                   top=top_n)

    # Step 4: Optional DELF reranking
    if use_rerank:
        confidence_df = delf_rerank(query=query_embedding,
                                    key=train_embeddings,
                                    query_image_index=0,
                                    confidence_df=confidence_df,
                                    re_sort=True)

    # Step 5: Predicted landmark ID
    predicted_landmark = confidence_df['prediction'].iloc[0]
    print(f"\nğŸ“� Predicted Landmark ID: {predicted_landmark}")

    # Step 6: Show query image
    query_image = get_image(query_image_path, resize=True, target_size=DELF_IMG_SIZE)
    plt.figure(figsize=(4, 4))
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis("off")
    plt.show()

    # Step 7: Show top-N similar images
    fig, axs = plt.subplots(1, top_n, figsize=(15, 5))
    for i in range(top_n):
        img = get_image(confidence_df['image_paths'].iloc[i], resize=True, target_size=DELF_IMG_SIZE)
        axs[i].imshow(img)
        axs[i].set_title(f"ID: {confidence_df['prediction'].iloc[i]}\nSim: {confidence_df['cos_similarity'].iloc[i]:.2f}")
        axs[i].axis('off')
    plt.suptitle("Top Similar Images")
    plt.show()

    return confidence_df



results = recognize_and_visualize(
    query_image_path="/kaggle/input/test-landmark-images/eiffel.jpg",
    train_embeddings=train_embeddings,
    model=embedding_model,
    top_n=10,
    use_rerank=True
)


import os
print(os.path.exists("/kaggle/working/test_sub/24/51b1affeeaab36b3.jpg"))



embedding_model.save("efficientnet_embedding_model.h5")


import pickle

with open("train_embeddings.pkl", "wb") as f:
    pickle.dump(train_embeddings, f)


from tensorflow.keras.models import load_model
embedding_model = load_model("efficientnet_embedding_model.h5", compile=False)


import os
os.listdir("/kaggle/working")


import zipfile

# Create a zip file containing both model and embeddings
with zipfile.ZipFile("/kaggle/working/landmark_model_files.zip", "w") as zipf:
    zipf.write("/kaggle/working/efficientnet_embedding_model.h5", arcname="efficientnet_embedding_model.h5")
    zipf.write("/kaggle/working/train_embeddings.pkl", arcname="train_embeddings.pkl")


