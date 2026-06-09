# General
import pandas as pd
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
# For image/data preprocessing
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# For modeling, eval
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
# Performance
import tqdm as tqdm
import multiprocessing
from multiprocessing import Pool
# Weights & Biases
import wandb
from wandb.keras import WandbCallback
from kaggle_secrets import UserSecretsClient


# GPU ACCELERATOR ENABLED
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
else:
    print('Found GPU at: {}'.format(device_name))


multiprocessing.cpu_count()  


TRAIN_DIR = "../input/herbarium-2022-fgvc9/train_images/"
TEST_DIR = "../input/herbarium-2022-fgvc9/test_images/"

with open("../input/herbarium-2022-fgvc9/train_metadata.json") as json_file:
    train_meta = json.load(json_file)
with open("../input/herbarium-2022-fgvc9/test_metadata.json") as json_file:
    test_meta = json.load(json_file)


sample_sub = pd.read_csv("../input/herbarium-2022-fgvc9/sample_submission.csv")
print("SAMPLE SUBMISSION")
sample_sub.head()


df_test = pd.DataFrame(test_meta)
print("TEST DATA")
df_test.head() ##untouched until ready to predict


#Create a meta-data df that can be used to call in training images
ids = []
categories = []
paths = []

for annotation, image in zip(train_meta['annotations'], train_meta['images']):
    ids.append(image["image_id"])
    categories.append(annotation['category_id'])
    paths.append(image["file_name"])

df_train = pd.DataFrame({"id":ids, "category":categories, "path":paths})
df_train.head()


##extract metadata features by category to merge with df_meta
sci_name = {cat["category_id"]:cat["scientificName"] for cat in train_meta['categories']}
family = {cat["category_id"]:cat["family"] for cat in train_meta['categories']}
genus = {cat["category_id"]:cat["genus"] for cat in train_meta['categories']}
species = {cat["category_id"]:cat["species"] for cat in train_meta['categories']}

df_train["scientific_name"] = df_train["category"].map(sci_name)
df_train["family"] = df_train["category"].map(family)
df_train["genus"] = df_train["category"].map(genus)
df_train["species"] = df_train["category"].map(species)

##split the path based on '/' into parent and child folder. 
##lambda fn is applied to each row in the column to split each path
df_train['path'].apply(lambda x : x.split('/'))

#add categories/sub_categories equivalents to df_meta
df_train['parent_folder'] = df_train['path'].apply(lambda x : x.split('/')[0])
df_train['child_folder'] = df_train['path'].apply(lambda x : x.split('/')[1])


df_train.head()


def plot_random_images(metadata, directory, n_imgs, dims=[3,4], random_seed=12):
    """
    Function randomly selects paths from the train metadata and plots the corresponding image.  
    """
    np.random.seed = random_seed
    # Randomly sample n rows from metatdata
    rndm_elems = metadata.sample(n=n_imgs)      
    
    # Add the img path, category, and sci name to lists
    imgs = []
    category = []
    scientific_name = []
    for path, categ, sci_name in zip(rndm_elems['path'], rndm_elems['category'], rndm_elems['scientific_name']):
        imgs.append(cv2.imread(os.path.join(directory,path)))
        category.append(categ)
        scientific_name.append(sci_name)
    # Prepare figures/axes for subplots
    fig, axes = plt.subplots(dims[0], dims[1], figsize=(10,10))
    axes = axes.flatten()
    # For each image, plot image to a subplot and title with category + scientific name
    for img, ax, c, s in zip(imgs, axes, category, scientific_name):
        title = str(c) + " | " + s
        ax.imshow(img)
        ax.axis('off')
        ax.title.set_text(title)
    plt.suptitle("Example Images")
    plt.show()

    
TRAIN_DIR = "../input/herbarium-2022-fgvc9/train_images/"
plot_random_images(df_train, TRAIN_DIR, 8, [4,2], random_seed=123)


print("Parent_folders", df_train['parent_folder'].unique())
print("Child folders", df_train['child_folder'].unique())

parent_cats = str(df_train['category'].unique())
print("Categories", parent_cats)


cat_val_cnt = df_train['category'].value_counts()
print("Category, Number of Images: \n",
      cat_val_cnt) 
print("")
print("Number of categories which have 80 training images:", len(cat_val_cnt[cat_val_cnt == 80]))
cat_index = cat_val_cnt[cat_val_cnt == 80].sort_values(ascending=False).index


# Reduce training df to include only samples belonging to 80-image categories
red_train_df = df_train[df_train.category.isin(cat_index)]
red_train_df.info()


print("Now we have ", red_train_df['category'].nunique(), "categories instead of ", df_train['category'].nunique())
print("reduced training metadata shape: ", red_train_df.shape)


# Prep meta-data df for the flow_from_dataframe() fn to work properly
red_train_df = red_train_df.reset_index()
## Convert 'category' to string, which is required for categorical classes in flow_from_dataframe 
red_train_df['str_category'] = ''
red_train_df[['category']] = red_train_df[['category']].astype(int)
red_train_df[['str_category']] = red_train_df[['category']].astype(str)
red_train_df.head()


labels = red_train_df['str_category'].unique()
len(labels)


##Encode target labels with value between 0 and n_classes-1 (since the categories are currently an assortment of #s from all over the place)
le = LabelEncoder() ##sklearn.preprocessing
encoded_labels = le.fit_transform(red_train_df['str_category'])
red_train_df['encoded_labels'] = encoded_labels
red_train_df[['encoded_labels']] = red_train_df[['encoded_labels']].astype(str) ##convert to string for flow_from_dataframe()


num_classes = 174 #red_train_df['category'].nunique()
f1_macro = tfa.metrics.F1Score(num_classes=num_classes, average='macro') ##from TensorFlow Addons


# Default float type is 32
print("default float type:", tf.keras.backend.floatx() )
# Reduce to mixed precision: https://www.tensorflow.org/api_docs/python/tf/keras/backend/set_floatx
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


wandb.login(key='fed8886fa715351293a079cd945d36b6baa126db')


# Param dict
default=dict(
    dropout = 0.35,
    kernel_size=(3,3),
    layer_1_size = 32,
    layer_2_size = 32,
    pool_1_size = (2,2),
    layer_3_size=64,
    layer_4_size=64,
    pool_2_size=(3,3),
    layer_5_size=1024,
    layer_6_size=420,
    learn_rate = 0.001,
    beta_1 = 0.9,
    beta_2 = 0.999,
    epochs = 100,
    batch_size = 64,
    img_size = (120, 120),
    architecture="CNN",
    infra="Kaggle"
   )

# Weights & Biases Initialization
wandb.init(anonymous='allow', project="herb22", config=default)
config = wandb.config


# 2d ConvNet
img_shape = (120, 120, 3)
cnnmod = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=img_shape),
    tf.keras.layers.Conv2D(filters=config.layer_1_size, kernel_size=config.kernel_size, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=config.layer_2_size, kernel_size=config.kernel_size, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=config.pool_1_size),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(config.dropout),
    tf.keras.layers.Conv2D(filters=config.layer_3_size, kernel_size=config.kernel_size, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=config.layer_4_size, kernel_size=config.kernel_size, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=config.pool_2_size),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(config.dropout),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(config.layer_5_size, activation='relu'),
    tf.keras.layers.Dense(config.layer_6_size, activation='relu'),
    tf.keras.layers.Dense(174, activation='softmax')
])

cnnmod.summary()

cnnweights = cnnmod.get_weights()
#fn to reset model weights to randomly initialized if want to restart training
reset_model = lambda model, weights: model.set_weights(weights) 
# ------------------------------------------------------------------------------

# Loss
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# Optimizier (adam)
optim = tf.keras.optimizers.Adam(learning_rate=config.learn_rate,
                                 beta_1=config.beta_1,
                                 beta_2=config.beta_2)

# Compile
cnnmod.compile(optimizer=optim,
               loss=loss_fn,
               metrics=['accuracy', f1_macro])

# -------------------------------------------------------------------------------
# Data Generator (for augmentation/preprocessing in the flow of training)

## Custom preprocessing function to apply to each image
def _adjust_image(img):
    """
    Uses cv2 to apply some preprocessing to each image to improve the data.
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  ##convert from RGB to YUV
    img_gray = img_yuv[:,:,0].astype(np.uint8) ##convert to single channel (Y channel is the luminance component)
    img_equ = cv2.equalizeHist(img_gray)       ##equalize histogram (note that only works for single channel unit8)
    img_yuv[:,:,0] = img_equ                   ##add equalized channel back in
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) ##convert back to RGB  
    return img_rgb

## ImageDataGenerator (train & validation)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=_adjust_image,
                rescale=1.0/255,
                rotation_range=30,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='reflect',
                 #cval = 235,   ##a bright constant value for the fill mode "constant"
                validation_split=0.2,
)


# Example Image
path = "../input/herbarium-2022-fgvc9/train_images/021/25/02125__015.jpg"
ex_img = cv2.imread(os.path.join(path))
plt.imshow(ex_img.astype(np.uint8))
plt.title("Example image (unprocessed)")
plt.show()


# Image after augmentation, preprocessing is applied
x = ex_img
x = x.reshape((1,) + x.shape)
aug_x = train_datagen.flow(x)
aug_images = [next(aug_x)[0] for i in range(12)]

fig, axes = plt.subplots(3,4,figsize=(10,10)) ##3 img rows, 4 img cols
axes = axes.flatten()
for img, ax in zip(aug_images,axes): ##zip image to its subplot
    ax.imshow(img)
    #ax.axis('off')
    
plt.suptitle("Example image after augmentation",fontsize=24)
plt.show()


reset_model(cnnmod, cnnweights) ##restore weights to random
epochs = config.epochs
batch_size = config.batch_size
target_size = config.img_size
labels = red_train_df['str_category'].unique()

# TRAIN
history_cnn = cnnmod.fit(train_datagen.flow_from_dataframe(dataframe=red_train_df,
                                                                    directory="../input/herbarium-2022-fgvc9/train_images",
                                                                    x_col='path',
                                                                    y_col='encoded_labels',
                                                                    target_size = target_size,
                                                                    batch_size = batch_size,
                                                                    subset = "training"
                                                                    ),
                                   validation_data = train_datagen.flow_from_dataframe(dataframe=red_train_df,
                                                                    directory="../input/herbarium-2022-fgvc9/train_images",
                                                                    x_col='path',
                                                                    y_col='encoded_labels',
                                                                    target_size = target_size,
                                                                    batch_size = batch_size,
                                                                    subset = "validation"
                                                                    ),
                                   epochs = epochs,
                                   callbacks=[WandbCallback(input_type="images", labels=labels)]
                                  )


wandb.finish() ##tell wandb to stop tracking the session


## Plotting the fit history
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,10))

ax[0].plot(history_cnn.history['loss'], label='loss')
ax[0].plot(history_cnn.history['val_loss'], label='val loss')
ax[0].legend()

ax[1].plot(history_cnn.history['accuracy'], label='acc')
ax[1].plot(history_cnn.history['val_accuracy'], label='val acc')
ax[1].legend()

ax[2].plot(history_cnn.history['f1_score'], label='f1')
ax[2].plot(history_cnn.history['val_f1_score'], label='val f1')
ax[2].legend()

plt.show()


# Predict model onto test data
##Test-set datagen:
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                               preprocessing_function=_adjust_image
).flow_from_dataframe(
    dataframe=df_test,
    directory="../input/herbarium-2022-fgvc9/test_images/",
    x_col='file_name',
    y_col=None,
    class_mode=None,
    target_size=target_size,
    batch_size = 128,
    shuffle=False
)

## Predict with generator
y_pred = cnnmod.predict(test_datagen)


# Determine the (encoded) class with highest predicted probability
y_pred_encoded = np.argmax(y_pred, axis=1)

print("y_pred_encoded.shape ", y_pred_encoded.shape)
print("Model's unique class guesses (out of 174 possible): ", len(np.unique(y_pred_encoded)) )


# Reverse the label encodings to get true category labels
y_pred_class = le.inverse_transform(y_pred_encoded)


# Add predictions to a submission df with test-sample/image Id
submission = sample_sub.copy()
submission.drop(labels='Predicted',axis=1)
submission['Predicted'] = y_pred_class
submission.head()


# Convert to csv and save
submission.to_csv("submission.csv", index=False)

