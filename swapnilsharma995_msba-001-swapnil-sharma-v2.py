import os, zipfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg



with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/train')

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test')

shutil.copy('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv', '/kaggle/working/sample_submission.csv')


base_dir = '/kaggle/working/dataset'
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir + '/cats', exist_ok=True)
os.makedirs(train_dir + '/dogs', exist_ok=True)

for file in os.listdir('/kaggle/working/train/train'):
    src = f'/kaggle/working/train/train/{file}'
    dst = f'{train_dir}/cats/{file}' if 'cat' in file else f'{train_dir}/dogs/{file}'
    shutil.move(src, dst)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Define directories for plotting
cat_dir = os.path.join(train_dir, 'cats')
dog_dir = os.path.join(train_dir, 'dogs')

def plot_samples(image_dir, title, num_samples=5):
    images = random.sample(os.listdir(image_dir), num_samples)
    plt.figure(figsize=(15, 5))
    for i, img_file in enumerate(images):
        img_path = os.path.join(image_dir, img_file)
        img = mpimg.imread(img_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{title} {i+1}")
    plt.suptitle(f"Sample {title} Images", fontsize=16)
    plt.show()

# Show samples
plot_samples(cat_dir, 'Cat')
plot_samples(dog_dir, 'Dog')



# %%
img_size = 160
batch_size = 16

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


# %%
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# %% [markdown]
# ## 7. Model Training
# Train the model for a few epochs using the training and validation generators.

# %%
model.fit(train_generator, validation_data=val_generator, epochs=3)




# Prepare test data, generate predictions using the trained model.

# %%
test_dir = '/kaggle/working/test/test'
test_df = pd.DataFrame({
    'filename': sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]))
})
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col=None,
    class_mode=None,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False
)

predictions = model.predict(test_gen, verbose=1).flatten()




# Convert predictions to class labels: 0 for cat, 1 for dog
predicted_classes = (predictions > 0.5).astype(int)

# Map numeric labels to text
label_map = {0: 'Cat', 1: 'Dog'}

# Display a few test images with predicted labels
def show_predictions(test_df, predicted_classes, num_images=10):
    plt.figure(figsize=(20, 5))
    indices = np.random.choice(len(test_df), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        img_path = test_df.iloc[idx]['filepath']
        img = mpimg.imread(img_path)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis('off')
        pred_label = label_map[predicted_classes[idx]]
        plt.title(pred_label)
    plt.suptitle("Predicted Labels on Test Images", fontsize=18)
    plt.show()

show_predictions(test_df, predicted_classes, num_images=10)



# %%
submission = pd.read_csv('/kaggle/working/sample_submission.csv')
submission['label'] = predictions
submission.to_csv('/kaggle/working/MSBA.SessionX.SwapnilSharma.csv', index=False)

# Display a few predictions
print("âœ… Lightweight submission saved.")
print(submission.head())

