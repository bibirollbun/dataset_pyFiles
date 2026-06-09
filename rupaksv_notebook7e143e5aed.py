import os

base = "/kaggle/input"
file_list = []
for dirname, _, filenames in os.walk(base):
    for filename in filenames:
        file_list.append(os.path.join(dirname, filename))

print("Total files found:", len(file_list))
print("First 10 files:", file_list[:10])


import pandas as pd
import numpy as np


train_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')


train_df.info()


train_df.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


IMG_DIR = "/kaggle/input/grand-xray-slam-division-a/train1"
for i in range(5):
    row = train_df.iloc[i]
    path = os.path.join(IMG_DIR, row["Image_name"])
    img = mpimg.imread(path)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {row['Patient_ID']}")
    plt.axis("off")
    plt.show()



from PIL import Image
img_path = os.path.join(IMG_DIR, os.listdir(IMG_DIR)[0])
print("Sample path:", img_path)

# open and check size
img = Image.open(img_path)
print("Image size (W,H):", img.size)
print("Mode:", img.mode)


# IMG_DIR = "/kaggle/input/grand-xray-slam-division-a/train1"

# # Choose the series ID you want to see
# series_id = "00000011"

# # Get all files with that series id
# series_files = [f for f in os.listdir(IMG_DIR) if f.split("_")[0] == series_id]

# print(f"Found {len(series_files)} images for series {series_id}")

# # Plot them
# plt.figure(figsize=(20, 10))
# for i, fname in enumerate(series_files):
#     path = os.path.join(IMG_DIR, fname)
#     img = Image.open(path)
    
#     plt.subplot(1, len(series_files), i+1)
#     plt.imshow(img, cmap="gray")
#     plt.axis("off")
#     plt.title(fname)

# plt.show()


# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# # Path to a sample grayscale X-ray
# img_path = "/kaggle/input/grand-xray-slam-division-a/train1/00000011_013_001.jpg"

# # Open grayscale image
# img_gray = Image.open(img_path).convert("L")  # ensure grayscale

# # Convert grayscale to RGB by repeating the channel
# img_rgb = img_gray.convert("RGB")  # or np.stack([np.array(img_gray)]*3, axis=-1)

# # Plot both images side by side
# plt.figure(figsize=(10,5))

# plt.subplot(1,2,1)
# plt.imshow(img_gray, cmap="gray")
# plt.title("Original Grayscale")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(img_rgb)
# plt.title("Converted to RGB")
# plt.axis("off")

# plt.show()



import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50


base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top = False,
    weights = None
)


from tensorflow.keras import layers, models
base_model.trainable = False
x = base_model.get_layer("conv4_block4_out").output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(500,activation ='relu')(x)
x = layers.Dense(250,activation ='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(14,activation ='sigmoid')(x)

model  = tf.keras.Model(inputs =base_model.input,outputs=output)


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',    
    patience=3,            
    restore_best_weights=True, 
    min_delta=1e-4,        
    verbose=1
)


from tensorflow.keras.metrics import AUC
model.compile(optimizer= tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=[AUC(name='auc', multi_label=True, num_labels= 14)]
             )


img_list = [img for img in train_df['Image_name']]
img_path  = [os.path.join(IMG_DIR,img) for img in train_df['Image_name']]
target = train_df[train_df.columns[7:]].values


IMG_SIZE = (224,224)
def pre_process(path, label):
    img = tf.io.read_file(path)
    
    img = tf.image.decode_jpeg(img, channels=3)
    
    img = tf.image.resize(img, IMG_SIZE)
    
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    label = tf.cast(label, tf.float32)
    
    return img, label


from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    img_path, target, test_size=0.1, random_state=42
)
dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
dataset = dataset.map(pre_process, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(10000).batch(1000).prefetch(tf.data.AUTOTUNE)



val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(pre_process, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(1000).prefetch(tf.data.AUTOTUNE)


#checkpoint_path = "/kaggle/working/resnet50_multi_label.weights.h5"
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True,
#     verbose=1
# )
history = model.fit(
    dataset,
    validation_data = val_dataset,
    epochs = 10,
    callbacks=[early_stop]
)


print(history.history.keys())


import matplotlib.pyplot as plt

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot AUC
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('Training and Validation AUC')
plt.legend()
plt.show()



submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')


test_dir = "/kaggle/input/grand-xray-slam-division-a/test1"
test_paths = [os.path.join(test_dir, fname) for fname in submission['Image_name']]


IMG_SIZE = (224, 224)
def preprocess_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)


preds = model.predict(test_ds, verbose=1)


threshold = 0.5
preds_binary = (preds >= threshold).astype(int)

submission.iloc[:, 1:] = preds_binary


submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission saved:", submission.shape)




