import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
train_df['path'] = '/kaggle/input/petfinder-pawpularity-score/train/' + train_df['Id'] + '.jpg'

meta_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group','Collage','Human','Occlusion','Info','Blur'
]


scaler = StandardScaler()
X_meta = scaler.fit_transform(train_df[meta_features].values)


train_df, val_df, X_train_meta, X_val_meta = train_test_split(train_df, X_meta, test_size=0.2, random_state=42)



# =============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen_train = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
datagen_val = ImageDataGenerator(rescale=1./255)

train_gen = datagen_train.flow_from_dataframe(
    train_df, x_col='path', y_col='Pawpularity',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=True
)

val_gen = datagen_val.flow_from_dataframe(
    val_df, x_col='path', y_col='Pawpularity',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
)


def combined_generator(image_gen, meta_data):
    while True:
        img_batch, y_batch = next(image_gen)
        batch_size = img_batch.shape[0]
        meta_batch = meta_data[:batch_size]  # slice metadata
        yield {"image_input": img_batch, "meta_input": meta_batch}, y_batch



image_input = Input(shape=(224,224,3), name='image_input')
meta_input = Input(shape=(len(meta_features),), name='meta_input')



base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=image_input)
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)


base_model.load_weights("/kaggle/input/efficientnet-weights/efficientNetb0_weights.weights.h5")


m = Dense(64, activation='relu')(meta_input)
m = Dropout(0.3)(m)
m = Dense(32, activation='relu')(m)


# Combine
combined = Concatenate()([x, m])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.4)(z)
output = Dense(1, activation='linear')(z)

model = Model(inputs=[image_input, meta_input], outputs=output)
model.compile(optimizer=Adam(1e-4), loss='mean_squared_error', metrics=['RootMeanSquaredError'])
from tensorflow.keras.callbacks import EarlyStopping
# model.summary()


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_RootMeanSquaredError',  # ðŸ‘ˆ exact name from the logs
    mode='min',
    patience=3,
    restore_best_weights=True,
    verbose=1
)



history = model.fit(
    combined_generator(train_gen, X_train_meta),
    steps_per_epoch=len(train_gen),
    validation_data=combined_generator(val_gen, X_val_meta),
    validation_steps=len(val_gen),
    epochs=10,
    callbacks=[early_stop]
)



import matplotlib.pyplot as plt

# Plot RMSE
plt.figure(figsize=(8, 4))
plt.plot(history.history['RootMeanSquaredError'], label='Train RMSE')
plt.plot(history.history['val_RootMeanSquaredError'], label='Val RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Training vs Validation RMSE')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()



import pandas as pd
import os

# Path to your dataset CSV and image folder
csv_path = '/kaggle/input/petfinder-pawpularity-score/train.csv'   # or your local path
image_dir = '/kaggle/input/petfinder-pawpularity-score/train/'    # or your local path

# Load the dataset
df = pd.read_csv(csv_path)

# Add the image path column
df['path'] = df['Id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Quick check
print(df.head())



meta_features = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
                 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion',
                 'Info', 'Blur']

print(df[meta_features].head())



from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)



X_train_meta = train_df[meta_features].values
X_val_meta = val_df[meta_features].values
X_test_meta = test_df[meta_features].values

y_train = train_df['Pawpularity'].values
y_val = val_df['Pawpularity'].values
y_test = test_df['Pawpularity'].values



train_df


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerators for train, validation, and test
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size and image size
batch_size = 32
image_size = (224, 224)

# Train generator
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='Pawpularity',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='raw',
    shuffle=True
)

# Validation generator
val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='Pawpularity',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

# Test generator
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='Pawpularity',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)



import numpy as np

def combined_generator(image_gen, meta_data):
    while True:
        img_batch, y_batch = next(image_gen)
        idx = image_gen.index_array if image_gen.shuffle else np.arange(len(meta_data))
        meta_batch = meta_data[idx[:len(img_batch)]]
        yield ({'image_input': img_batch, 'meta_input': meta_batch}, y_batch)





test_loss, test_rmse = model.evaluate(
    combined_generator(test_gen, X_test_meta),
    steps=len(test_gen)
)
print(f"âœ… Test RMSE: {test_rmse:.4f}")



# import numpy as np

# def combined_generator(img_gen, meta_array):
#     batch_size = img_gen.batch_size
#     while True:
#         for i, (images, _) in enumerate(img_gen):  # unpack tuple; ignore labels
#             start_idx = i * batch_size
#             end_idx = start_idx + images.shape[0]  # handle last batch
#             meta_batch = meta_array[start_idx:end_idx]
#             yield [images, meta_batch]
def combined_generator():
    for batch_idx, batch in enumerate(test_gen_1):
        # Get images
        images = batch[0] if isinstance(batch, tuple) else batch
        
        # Slice metadata
        start_idx = batch_idx * test_gen_1.batch_size
        end_idx = start_idx + images.shape[0]
        meta_batch = X_test_meta[start_idx:end_idx]
        
        # Convert to tf.Tensor
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        meta_batch = tf.convert_to_tensor(meta_batch, dtype=tf.float32)
        
        # Yield tuple of tensors
        yield (images, meta_batch)



test_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')
test_df


import pandas as pd
import os

# Path to your dataset CSV and image folder
csv_path = '/kaggle/input/petfinder-pawpularity-score/test.csv'   # or your local path
image_dir = '/kaggle/input/petfinder-pawpularity-score/test/'    # or your local path

# Load the dataset
test_df = pd.read_csv(csv_path)

# Add the image path column
test_df['path'] = test_df['Id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Quick check
print(test_df.head())



test_gen_1 = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col=None,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)


meta_features = ['Subject Focus','Eyes','Face','Near','Action','Accessory',
                 'Group','Collage','Human','Occlusion','Info','Blur']

X_test_meta = test_df[meta_features].values
X_test_meta = scaler.transform(X_test_meta)


output_signature = (
    tf.TensorSpec(shape=(None, *test_gen_1.image_shape), dtype=tf.float32),  # images
    tf.TensorSpec(shape=(None, X_test_meta.shape[1]), dtype=tf.float32)      # metadata
)



import tensorflow as tf

# Ensure X_test_images and X_test_meta are numpy arrays
X_test_images = np.array([batch[0] if isinstance(batch, tuple) else batch for batch in test_gen_1])
X_test_meta   = np.array(X_test_meta)

# Create a tf.data.Dataset directly
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_images, X_test_meta))
test_dataset = test_dataset.batch(test_gen_1.batch_size)  # optional batching


test_dataset = tf.data.Dataset.from_generator(
    combined_generator,
    output_signature=output_signature
)



preds = model.predict(test_dataset, steps=len(test_gen_1), verbose=1)
print("Predictions shape:", preds.shape)
print("Sample predictions:", preds[:5].flatten())



import pandas as pd
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Pawpularity': preds.flatten()
})


submission


submission.to_csv('/kaggle/working/submission.csv', index=False)



s = pd.read_csv("submission.csv")
s


import pandas as pd
df = pd.read_csv("/kaggle/working/submission.csv")
print(df.columns)




