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


!pip install keras-tuner
!pip install scikeras


import matplotlib.pyplot as plt
import tensorflow as tf

import missingno as msno
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU!")
except:
    strategy = tf.distribute.get_strategy()  # Default to GPU/CPU
    print("Running on CPU/GPU")


class Config():
    BATCH_SIZE = 32
    IMG_SIZE = (96, 96)
    SEED = 42


tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)


# For matplotlib
plt.rcParams['axes.facecolor'] = '#F9FAFB'
plt.rcParams['figure.facecolor'] = '#F9FAFB'
plt.rcParams['axes.edgecolor'] = '#E5E7EB'
plt.rcParams['axes.labelcolor'] = '#111827'
plt.rcParams['xtick.color'] = '#6B7280'
plt.rcParams['ytick.color'] = '#6B7280'
plt.rcParams['text.color'] = '#111827'

# Custom colors for keypoints plot
keypoint_color = '#EF4444'
highlight_color = '#14B8A6'


train_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip', compression='zip')
test_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip', compression='zip')
lookup_df = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv',header=0, sep=',', quotechar='"')
submission_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/SampleSubmission.csv')


train_df.head()


test_df.head()


lookup_df.head()


train_df.shape


train_df.info()


train_df.isnull().sum()


msno.matrix(train_df)


img = np.array(train_df['Image'][0].split(), dtype='float32').reshape(*Config.IMG_SIZE)
plt.hist(img.ravel(), bins=50)


eye_cols = [col for col in train_df.columns if '_eye_' in col]
sns.histplot(data=train_df[eye_cols].melt(), x='value', hue='variable', kde=True, bins=50)


eyebrow_cols = [col for col in train_df.columns if '_eyebrow_' in col]
sns.histplot(data=train_df[eyebrow_cols].melt(), x='value', hue='variable', kde=True)


nose_cols = [col for col in train_df.columns if 'nose' in col]
sns.histplot(data=train_df[nose_cols].melt(), x='value', hue='variable', kde=True)


mouth_cols = [col for col in train_df.columns if 'mouth' in col]
sns.histplot(data=train_df[mouth_cols].melt(), x='value', hue='variable', kde=True)


corr = train_df.drop(columns=['Image']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Keypoint Correlation Heatmap")
plt.show()


def show_keypoints(image, keypoints):
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[0::2], keypoints[1::2], c='red', s=10)

def visualize_sample(df, index):
    row = df.iloc[index]
    image = np.array(row['Image'].split(), dtype='float32').reshape(*Config.IMG_SIZE)
    keypoints = row[:-1].values
    show_keypoints(image, keypoints)


train_df_clean = train_df.dropna()
visualize_sample(train_df_clean, 8)


train_df_with_inner_eye_na = train_df[train_df['left_eye_inner_corner_x'].isna()]

visualize_sample(train_df_with_inner_eye_na, 8)


train_df.fillna(method = 'ffill', inplace=True)


train_df.isnull().sum().any()


# convert image column to 96x96 float32 arrays
def process_image(img_str):
    img = np.array(img_str.split(), dtype=np.float32).reshape(Config.IMG_SIZE)
    img = img / 255.0  # normalize to [0,1]
    return img

train_df['Image'] = train_df['Image'].apply(process_image)


# extract inputs and labels
X = np.stack(train_df['Image'].values)
X = np.expand_dims(X, axis=-1)  # shape: (n_samples, 96, 96, 1)

y = train_df.drop(columns=['Image']).values.astype(np.float32)  # shape: (n_samples, 30)


def augment_image(image, keypoints):
    # Apply random horizontal flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        # Flip X coordinates: x_new = 96 - x
        keypoints = tf.concat([96 - keypoints[..., ::2], keypoints[..., 1::2]], axis=-1)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    return image, keypoints

def augment_image_with_mask(image, keypoints, mask):
    image, keypoints = augment_image(image, keypoints)
    return image, keypoints, mask


# Build a tf.data pipeline
def create_tf_dataset(X, y, mask=None, batch_size=Config.BATCH_SIZE, augment=False):
    if mask is not None:
        dataset = tf.data.Dataset.from_tensor_slices((X, y, mask))
        if augment:
            dataset = dataset.map(augment_image_with_mask, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if augment:
            dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


total_samples = len(X)
val_size = int(0.2 * total_samples)

train_X, val_X = X[val_size:], X[:val_size]
train_y, val_y = y[val_size:], y[:val_size]

train_ds = create_tf_dataset(train_X, train_y, mask=None, augment=True)
val_ds = create_tf_dataset(val_X, val_y, mask=None, augment=False)


def build_cnn_1_model(hp):
    model = Sequential([
        Input(shape=(*Config.IMG_SIZE, 1)),
        
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(hp.Int("conv_units", 64, 128, step=64), (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        GlobalAveragePooling2D(),
        Dense(hp.Int("dense_units", 128, 256, step=128), activation='relu'),
        Dropout(0.3),
        Dense(128,activation='relu'),
        Dropout(0.1),
        Dense(30)
])
    model.compile(optimizer=Adam(hp.Choice("learning_rate", [1e-4, 1e-3])),  
                                 metrics=[RootMeanSquaredError()], 
                                 loss='mse')
    return model

cnn_1_tuner = kt.RandomSearch(
    build_cnn_1_model,
    objective=kt.Objective("val_root_mean_squared_error", direction="min"),
    max_trials=50,
    directory="cnn_tuning_2"
)

cnn_1_tuner.search(X, y, 
                   epochs=10, 
                   validation_split=0.2)


cnn_1_tuner.results_summary()


cnn_1_model = cnn_1_tuner.get_best_models(num_models=1)[0]
cnn_1_model.summary()


EYE_NOSE_POINTS = [c for c in list(train_df.columns) if ('eye_' in c or 'nose' in c)]
EYE_NOSE_POINTS.append('Image')

MOUTH_EYEBROW_POINTS = [c for c in list(train_df.columns) if ('mouth' in c or 'eyebrow' in c)]
MOUTH_EYEBROW_POINTS.append('Image')


eye_nose_df = train_df[EYE_NOSE_POINTS]
X_eye_nose = np.stack(eye_nose_df['Image'].values)
X_eye_nose = np.expand_dims(X_eye_nose, axis=-1)  # shape: (n_samples, 96, 96, 1)

y_eye_nose = eye_nose_df.drop(columns=['Image']).values.astype(np.float32)  # shape: (n_samples, 30)


mouth_eyebrow_df = train_df[MOUTH_EYEBROW_POINTS]
X_mouth_eyebrow = np.stack(mouth_eyebrow_df['Image'].values)
X_mouth_eyebrow = np.expand_dims(X_mouth_eyebrow, axis=-1)  # shape: (n_samples, 96, 96, 1)

y_mouth_eyebrow = mouth_eyebrow_df.drop(columns=['Image']).values.astype(np.float32)  # shape: (n_samples, 30)


train_X_eye_nose, val_X_eye_nose = X_eye_nose[val_size:], X_eye_nose[:val_size]
train_y_eye_nose, val_y_eye_nose = y_eye_nose[val_size:], y_eye_nose[:val_size]

train_eye_nose_ds = create_tf_dataset(train_X_eye_nose, train_y_eye_nose, mask=None, augment=True)
val_eye_nose_ds = create_tf_dataset(val_X_eye_nose, val_y_eye_nose, mask=None, augment=False)


train_X_mouth_eyebrow, val_X_mouth_eyebrow = X_mouth_eyebrow[val_size:], X_mouth_eyebrow[:val_size]
train_y_mouth_eyebrow, val_y_mouth_eyebrow = y_mouth_eyebrow[val_size:], y_mouth_eyebrow[:val_size]

train_mouth_eyebrow_ds = create_tf_dataset(train_X_mouth_eyebrow, train_y_mouth_eyebrow, mask=None, augment=True)
val_mouth_eyebrow_ds = create_tf_dataset(val_X_mouth_eyebrow, val_y_mouth_eyebrow, mask=None, augment=False)


def build_eye_nose_model(hp):
    model = Sequential([
        Input(shape=(*Config.IMG_SIZE, 1)),

        Conv2D(hp.Int("conv_units_1", 32, 64, step=32), (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(hp.Int("conv_units_2", 64, 128, step=64), (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        GlobalAveragePooling2D(),
        Dense(hp.Int("dense_units", 128, 256, step=128), activation='relu'),
        Dropout(0.3),
        Dense(128,activation='relu'),
        Dropout(0.1),
        Dense(14)
])
    model.compile(optimizer=Adam(hp.Choice("learning_rate", [1e-4, 1e-3])),  
                                 metrics=[RootMeanSquaredError()], 
                                 loss='mse')
    return model


eye_nose_tuner = kt.RandomSearch(
    build_eye_nose_model,
    objective=kt.Objective("val_root_mean_squared_error", direction="min"),
    max_trials=50,
    directory="tuning_eye_nose"
)

eye_nose_tuner.search(X_eye_nose, 
                      y_eye_nose, 
                       epochs=10, 
                       validation_split=0.2)


def build_mouth_eyebrow_model(hp):
    model = Sequential([
        Input(shape=(*Config.IMG_SIZE, 1)),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(hp.Int("conv_units", 64, 128, step=64), (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        GlobalAveragePooling2D(),
        Dense(hp.Int("dense_units", 128, 256, step=128), activation='relu'),
        Dropout(0.3),
        Dense(128,activation='relu'),
        Dropout(0.1),
        Dense(16)
])
    model.compile(optimizer=Adam(hp.Choice("learning_rate", [1e-4, 1e-3])),  
                                 metrics=[RootMeanSquaredError()], 
                                 loss='mse')
    return model


mouth_eyebrow_tuner = kt.RandomSearch(
    build_mouth_eyebrow_model,
    objective=kt.Objective("val_root_mean_squared_error", direction="min"),
    max_trials=50,
    directory="tuning_mouth_eyebrow"
)

mouth_eyebrow_tuner.search(X_mouth_eyebrow, 
                                  y_mouth_eyebrow, 
                                  epochs=10, 
                                  validation_split=0.2)


eye_nose_tuner.results_summary()


mouth_eyebrow_tuner.results_summary()


best_eye_nose_model = eye_nose_tuner.get_best_models(1)[0]
best_eye_nose_model.summary()


best_mouth_eyebrow_model = mouth_eyebrow_tuner.get_best_models(1)[0]
best_mouth_eyebrow_model.summary()


def build_combined_model():
    inputs = Input(shape=(96, 96, 1))

    # Branch 1: block for eyes/nose
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(0.3)(x1)

    out1 = Dense(14, name="eye_nose")(x1)
    
    # Branch 2: block for mouth/eyebrows
    x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x2 = MaxPooling2D()(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D()(x2)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dropout(0.3)(x2)
    
    out2 = Dense(16, name="mouth_eyebrow")(x2)
    
    # Combine outputs
    outputs = concatenate([out1, out2], name='combined_output')

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        metrics=[RootMeanSquaredError()], 
        loss='mse'
    )
    
    return model


learning_rate_reduction = ReduceLROnPlateau(monitor='val_root_mean_squared_error', 
                                            factor=0.5, 
                                            patience=5, 
                                            min_lr=1e-7)

earlystop = EarlyStopping(monitor='val_root_mean_squared_error', 
                          patience=5, 
                          restore_best_weights=True)


basic_hps = cnn_1_tuner.get_best_hyperparameters(num_trials=1)[0]

basic_model = build_cnn_1_model(basic_hps)

basic_history = basic_model.fit(X, y,
                                  epochs=100, 
                                  validation_split=0.2, 
                                  callbacks=[learning_rate_reduction, earlystop])


region_model = build_combined_model()
region_history = region_model.fit(X, y,
                                  epochs=100, 
                                  validation_split=0.2, 
                                  callbacks=[learning_rate_reduction, earlystop])


def plot_history(history):
    plt.figure(figsize=(12, 5))

    # --- Plot Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # --- Plot RMSE ---
    plt.subplot(1, 2, 2)
    plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
    plt.title('RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_history(basic_history)


plot_history(region_history)


test_df['Image'] = test_df['Image'].apply(process_image)


def create_test_tf_dataset(X, batch_size=Config.BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


X_test = np.stack(test_df['Image'].values)
X_test = np.expand_dims(X_test, axis=-1)  # Shape: (n_samples, 96, 96, 1)

test_ds = create_test_tf_dataset(X_test)


predictions_basic = basic_model.predict(X_test)
predictions_region = region_model.predict(X_test)


feature_names = train_df.columns[:-1]


submission_basic = lookup_df.copy()

for i, row in submission_basic.iterrows():
    image_index = int(row['ImageId']) - 1  # ImageId starts at 1
    feature_name = row['FeatureName']
    feature_idx = feature_names.get_loc(feature_name)

    # Get the predicted value
    pred_value = predictions_basic[image_index][feature_idx]

    # Fill in Location
    submission_basic.at[i, 'Location'] = pred_value


submission_basic['Location'] = submission_basic['Location'].clip(0, 96)


submission_basic.head()


def show_image_with_keypoints(image, keypoints, title=None):
    """
    Show 96x96 image with 15 (x, y) keypoint pairs.
    """
    plt.imshow(image.squeeze(), cmap='gray')
    
    # Plot x and y in pairs
    x_points = keypoints[0::2]
    y_points = keypoints[1::2]
    
    plt.scatter(x_points, y_points, c='red', s=20)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


index = 10

# Get image and prediction
img = X_test[index]
pred = predictions_basic[index]

# Show
show_image_with_keypoints(img, pred, title=f"Test Image #{index}")


submission_basic[['RowId', 'Location']].to_csv('submission_basic.csv', index=False)


submission_region = lookup_df.copy()

for i, row in submission_region.iterrows():
    image_index = int(row['ImageId']) - 1  # ImageId starts at 1
    feature_name = row['FeatureName']
    feature_idx = feature_names.get_loc(feature_name)

    # Get the predicted value
    pred_value = predictions_region[image_index][feature_idx]

    # Fill in Location
    submission_region.at[i, 'Location'] = pred_value


submission_region['Location'] = submission_region['Location'].clip(0, 96)


submission_region.head()


index = 10

# Get image and prediction
img = X_test[index]
pred = predictions_region[index]

# Show
show_image_with_keypoints(img, pred, title=f"Test Image #{index}")


submission_region[['RowId', 'Location']].to_csv('submission_region.csv', index=False)

