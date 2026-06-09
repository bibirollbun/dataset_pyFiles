# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras as ks 


# import dataset
df = pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')
df.head()


# check the shape of the dataset
print(f'{df.shape[0]} rows and {df.shape[1]} columns')


df['label'].unique().tolist()


df['age'].describe()


# Plot the data count based on variety name
df['variety'].value_counts().plot(kind='bar')
plt.show()


# Plot the data count based on variety name
df['label'].value_counts().plot(kind='bar')
plt.show()


normal = df[df['label'] == 'normal']
normal= normal[normal['variety'] == 'ADT45']    
five_normals = normal.image_id[18:24].values
print(five_normals.tolist())

dead = df[df['label'] == 'dead_heart']
normal= dead[dead['variety'] == 'ADT45']    
five_deads = dead.image_id[18:24].values
print(five_deads.tolist())


# Make plot of images just to have an idea
plt.figure(figsize=(20,10))
cols = 6
path = '/kaggle/input/paddy-disease-classification/train_images/'
for i, image_loc in enumerate(np.concatenate((five_normals, five_deads))):
    plt.subplot(10//cols + 1, cols, i + 1)

    if i < len(five_normals):
        img = plt.imread(path + 'normal/' + image_loc)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Normal')
    else:
        img = plt.imread(path + 'dead_heart/' + image_loc)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Dead')
    



images = [
    '/kaggle/input/paddy-disease-classification/train_images/hispa/106590.jpg', \
    '/kaggle/input/paddy-disease-classification/train_images/tungro/109629.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_blight/109372.jpg', \
    '/kaggle/input/paddy-disease-classification/train_images/downy_mildew/102350.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/blast/110243.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_streak/101104.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/normal/109760.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/brown_spot/104675.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/dead_heart/105159.jpg',\
    '/kaggle/input/paddy-disease-classification/train_images/bacterial_panicle_blight/101351.jpg',\
    ]

diseases = ['hispa', 'tungro', 'bacterial_leaf_blight', 'downy_mildew', 'blast', 'bacterial',
'normal','brown_spot', 'dead_heart', 'bacterial_panicle_blight']
diseases = [disease +' image' for disease in diseases]

plt.figure(figsize=(20,10)) 
columns = 5

for i, image_loc in enumerate( images):
    plt.subplot(len(images)//columns + 1,columns, i + 1)
    image=plt. imread (image_loc)
    plt. title(diseases[i])
    plt. imshow (image)


df.head()


# lets encode the label & variety columns
le_list = {}
for col in ['label', 'variety']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_list[col] = le


# define parameters
batch_size = 32
image_size = (224, 224)
num_classes = 10
epochs = 50


# split the data
train_df = ks.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset='training',
    seed=122,
    image_size=image_size,
    batch_size=batch_size
)
val_df = ks.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset='validation',
    seed=122,
    image_size=image_size,
    batch_size=batch_size
)


# get class names
class_names = train_df.class_names  
print(class_names)


# lets check the shape of img
for image_batch, labels_batch in train_df:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# lets normalize the data
normalization_layer = ks.layers.Rescaling(1./255)
normalized_df = train_df.map(lambda x, y: (normalization_layer(x), y))
first_image, first_label = next(iter(normalized_df))
# Check the pixel value range
print(first_image[0].numpy().min(), first_image[0].numpy().max())


AUTOTUNE = tf.data.AUTOTUNE
train_df = train_df.cache().prefetch(buffer_size=AUTOTUNE)
val_df = val_df.cache().prefetch(buffer_size=AUTOTUNE)


# create the model
model = ks.Sequential([
    ks.Input(shape=(224, 224, 3)),       # Correct input shape
    ks.layers.Rescaling(1./255),

    ks.layers.Conv2D(32, 3, padding='same', activation='relu'),
    ks.layers.MaxPooling2D(),

    ks.layers.Conv2D(32, 3, padding='same', activation='relu'),
    ks.layers.MaxPooling2D(),

    ks.layers.Conv2D(64, 3, padding='same', activation='relu'),
    ks.layers.MaxPooling2D(),

    ks.layers.Conv2D(128, 3, padding='same', activation='relu'),
    ks.layers.MaxPooling2D(),

    ks.layers.Flatten(),                        # This will now be 12544
    ks.layers.Dropout(0.2),
    
    ks.layers.Dense(256, activation='relu'),    # Will automatically fit 12544 input
    ks.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# train the model
history = model.fit(
    train_df,
    validation_data=val_df,
    epochs=20
)


# save the model
model.save('Rice_Disease_Classification_Model.keras')


# evaluate the model
loss, accuracy = model.evaluate(train_df)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')


import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# === CONFIG ===
TEST_DIR = '/kaggle/input/paddy-disease-classification/test_images/'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# MODEL_PATH = 'model.h5'  # path to your saved model

# === Load test filenames ===
test_filenames = os.listdir(TEST_DIR)

# Create DataFrame for generator
test_df = pd.DataFrame({'filename': test_filenames})

# === Prepare Test Data Generator ===
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_DIR,
    x_col='filename',
    y_col=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)



# === Predict on Test Data ===
preds = model.predict(test_generator, verbose=1)
predicted_class_indices = np.argmax(preds, axis=1)

# === Get Class Labels ===
# You need to have access to train_generator.class_indices
# Example: {'bacterial_leaf_blight': 0, 'brown_spot': 1, ...}
class_names = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

class_indices = {name: idx for idx, name in enumerate(class_names)}
labels = dict((v, k) for k, v in class_indices.items())

# Map predicted indices to class names
predicted_labels = [labels[k] for k in predicted_class_indices]

# === Create Submission File ===
submission = pd.DataFrame({
    'image_id': test_df['filename'],
    'label': predicted_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv generated successfully!")


