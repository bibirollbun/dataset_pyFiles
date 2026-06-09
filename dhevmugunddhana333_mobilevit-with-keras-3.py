# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
current_dir = "/kaggle/input/isic-2024-challenge"

# List all files in the current directory (excluding subdirectories)
files_in_dir = [f for f in os.listdir(current_dir)]
for file in files_in_dir:
    print(file)

    
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import h5py
import cv2
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
import plotly.express as px
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, mean_squared_error,get_scorer_names
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression,SGDClassifier
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import *
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.pipeline import Pipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import warnings
warnings.filterwarnings("ignore")


# Deep learning keras libraries
import keras
print(keras.__version__)
from keras import backend
from keras.losses import Loss
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from keras import ops
import keras_hub
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import math
from keras import ops
from keras import layers
from keras.saving import register_keras_serializable


# CONFIG VARIABLES AND PARAMETERS
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# DATA
BUFFER_SIZE = 512
BATCH_SIZE = 64

# Hyperparameters
PATCH_SIZE = 4  # 2x2, for the Transformer blocks.
IMAGE_SIZE = 128
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.
keras.utils.set_random_seed(48)


current_dir = "/kaggle/input/isic-2024-challenge"
sample_submission = pd.read_csv(os.path.join(current_dir,'sample_submission.csv'))
sample_submission.info()


sample_submission.head()


train_metadata = pd.read_csv(os.path.join(current_dir,'train-metadata.csv'),low_memory=False)
train_metadata.info()


train_metadata.head()


test_metadata = pd.read_csv(os.path.join(current_dir,'test-metadata.csv'),low_memory=False)
test_metadata.info()


test_metadata.head()


# image sample visualization
training_validation_hdf5 = h5py.File(f"{current_dir}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{current_dir}/test-image.hdf5", 'r')


# sample randomly two images from the train dataset
img_sample = train_metadata['isic_id'].sample(n=2).to_list()

# load the image from byte arrays, 
byte_str = [training_validation_hdf5[isic_id][()] for isic_id in img_sample]

# convert byte str to numpy array
img_arr = [np.frombuffer(byte, np.uint8) for byte in byte_str]

# convert cv2 image
img_cv2 = [cv2.imdecode(nparr, cv2.IMREAD_COLOR) for nparr in img_arr] 
for ind,val in enumerate(img_cv2):
    print(f"Image {img_sample[ind]}:")
    print(f"Shape:{val.shape}")
    cv2_imshow(val)


# sample randomly two images from the train dataset
img_sample = test_metadata['isic_id'].sample(n=3).to_list()

# load the image from byte arrays, 
byte_str = [testing_hdf5[isic_id][()] for isic_id in img_sample]

# convert byte str to numpy array
img_arr = [np.frombuffer(byte, np.uint8) for byte in byte_str]

# convert cv2 image
img_cv2 = [cv2.imdecode(nparr, cv2.IMREAD_COLOR) for nparr in img_arr] 
for ind,val in enumerate(img_cv2):
    print(f"Image {img_sample[ind]}:")
    print(f"Shape:{val.shape}")
    cv2_imshow(val)


class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.train_columns = None

    def fit_transform(self, df):
        """Preprocess training data and store transformations."""
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        df = self._drop_train_only_columns(df)
        categorical_cols, numerical_cols = self._identify_column_types(df)
        
        # Define preprocessing pipelines
        numerical_pipeline = Pipeline([
            ("imputer", KNNImputer()),
            ("scaler", StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        self.preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])
        
        transformed_data = self.preprocessor.fit_transform(df)
        
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
        all_columns = numerical_cols + list(cat_feature_names)
        
        df_processed = pd.DataFrame(transformed_data, columns=all_columns)
        df_processed["isic_id"] = df["isic_id"].values
        df_processed["target"] = df["target"].values
        
        self.train_columns = df_processed.columns  # Store train columns
        return df_processed

    def transform(self, df):
        """Preprocess test data using stored transformations from training."""
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        
        transformed_data = self.preprocessor.transform(df)
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out()
        all_columns = self.train_columns[:-2]  # Exclude 'isic_id' and 'target'
        
        df_processed = pd.DataFrame(transformed_data, columns=all_columns)
        df_processed["isic_id"] = df["isic_id"].values
        # df_processed["target"] = df["target"].values
        
        # Align test dataset with train columns
        df_processed = self._align_train_test_columns(df_processed)
        return df_processed

    def _drop_irrelevant_columns(self, df):
        """Remove unnecessary columns."""
        return df.drop(columns=['patient_id','image_type', 'tbp_tile_type', 'attribution', 'copyright_license'], errors="ignore")

    def _drop_train_only_columns(self,df):
        """Remove columns that are present only in the train set and not in the test set"""
        drop_train_only_columns = [
            'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
            'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence'
            ]
        return df.drop(columns=drop_train_only_columns,errors='ignore')
        

    def _identify_column_types(self, df):
        """Identify categorical and numerical columns, excluding 'isic_id' and 'target'."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != "isic_id"]
        numerical_cols = [col for col in numerical_cols if col != "target"]
        return categorical_cols, numerical_cols
    
    def _align_train_test_columns(self, df):
        """Ensure test data has the same columns as train data."""
        train_cols = list(self.train_columns)
        train_cols.remove('target')
        missing_cols = set(train_cols) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        return df[train_cols]



train_metadata = pd.read_csv(os.path.join(current_dir, 'train-metadata.csv'), low_memory=False)
test_metadata = pd.read_csv(os.path.join(current_dir, 'test-metadata.csv'), low_memory=False)


data_object = DataPreprocessor()
train_metadata_processed = data_object.fit_transform(train_metadata)
test_metadata_processed = data_object.transform(test_metadata)


X_train = train_metadata_processed.drop(columns=['target'])
y_train = train_metadata_processed['target']
X_test = test_metadata_processed.copy()


X_train_final = X_train.copy()
y_train_final = y_train.copy()


X_train_final.info()


y_train_final.info()


X_test.info()


X_test.head()


print("original_ratio:",y_train_final.value_counts()[1]/y_train_final.value_counts()[0])


def resampler_data(X, Y):
    
    # Apply undersampling only on numerical data
    resampler = RandomUnderSampler(sampling_strategy=0.01,)
    X_resampled, Y_resampled = resampler.fit_resample(X,Y)
    
    # now apply over sampling for the minorit class
    resampler = RandomOverSampler(sampling_strategy=0.05)
    X_resampled, Y_resampled = resampler.fit_resample(X_resampled, Y_resampled)

    print("X_final shape;",X_resampled.shape)
    print("Y_final shape:",Y_resampled.shape)
    
    return X_resampled, Y_resampled


X_train_final, y_train_final = resampler_data(X_train_final, y_train_final)


X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_final, y_train_final, test_size=0.1, stratify=y_train_final, random_state = 63
)


# Path to images
image_dir = '/kaggle/input/isic-2024-challenge/train-image/image'
# # Separate features and target
# tabular_features = tr_v2.drop(columns=['isic_id', 'target'])
# targets = tr_v2['target']


# # generating the train and val data

# def stratified_split(X, y, patient_col='patient_id', test_size=0.2, random_state=42):
#     """Create stratified split ensuring no patient_id overlap between sets"""
#     patient_targets = pd.DataFrame({
#         'patient_id': X[patient_col],
#         'target': y
#     }).drop_duplicates()
    
#     train_patients, val_patients = train_test_split(
#         patient_targets['patient_id'].unique(),
#         test_size=test_size,
#         random_state=random_state,
#         stratify=patient_targets.groupby('patient_id')['target'].first()
#     )
    
#     train_mask = X[patient_col].isin(train_patients)
#     val_mask = X[patient_col].isin(val_patients)
    
#     return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


# # Perform the split
# X_train_final, X_val, y_train_final, y_val = stratified_split(
#     X_train_resampled, Y_train_resampled
# )


X_train_final.info()


y_train_final.value_counts()


X_val.info()


y_val.value_counts()


# Image preprocessing function
def process_image(byte_array):
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, [IMAGE_HEIGHT,IMAGE_WIDTH])
    image = tf.io.decode_jpeg(byte_array, channels=3)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    # Image augmentation
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0, 1]
    return image

# Data preprocessing function
def preprocess_data(inp,targets):
    # Construct image path
    # def read_hdf5(id_tensor):
    #     # Convert tensor to string
    #     id_str = id_tensor.numpy().decode('utf-8')
    #     # Read from HDF5 and return the raw bytes
    #     return training_validation_hdf5[id_str][()].tobytes()
    
    # Wrap the HDF5 reading operation in py_function and specify the output type
    # byte_array = tf.py_function(
    #     read_hdf5,
    #     [isic_id],
    #     Tout=tf.string
    # )
    image = process_image(inp['images'])
    inp['images'] = image
    return (inp,targets)
    # return (image, target)

def load_image_bytes(isic_id):
    with h5py.File('/kaggle/input/isic-2024-challenge/train-image.hdf5', "r") as hdf5_file:
        return hdf5_file[isic_id][()].tobytes()


# Convert the dataframe into TensorFlow Dataset
def create_dataset(X,Y):
    # Convert tabular data to float32
    tabular_data = X.drop(columns=['isic_id']).values.astype('float32')
    
    # Ensure targets are in the correct format (assuming binary/categorical)
    targets = Y.values.astype('int32').reshape(-1,1)
    
    # Convert ISIC IDs to strings if they aren't already
    isic_ids = X['isic_id'].astype(str).values

    # # loading images as bytes from the hdf5 files before passing to dataset to improve efficiency and reduce computation time, 
    # imgs = [None]*len(isic_ids)
    # for i, isic_id in enumerate(tqdm(isic_ids, desc="Loading training Images ")):
    #     imgs[i] = training_validation_hdf5[isic_id][()].tobytes()

     # Use ProcessPoolExecutor for multiprocessing
    imgs = [None] * len(isic_ids)
    with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust workers based on CPU
        future_to_index = {executor.submit(load_image_bytes, isic_id): i for i, isic_id in enumerate(isic_ids)}
        for future in tqdm(as_completed(future_to_index), total=len(isic_ids), desc="Loading training images"):
            i = future_to_index[future]
            imgs[i] = future.result()
    
    inp = {"images": imgs, "tabular":tabular_data}
    slices = (inp,targets)
        
    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices(slices)
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset



# shuffle the train_dataset before batching
X_train_final, y_train_final = shuffle(X_train_final, y_train_final)
print(X_train_final.shape)
print(y_train_final.shape)


# train_dataset = create_dataset(train_dataframe).shuffle(len(train_dataframe)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_dataset = create_dataset(validation_dataframe).shuffle(len(validation_dataframe)).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
train_dataset = create_dataset(X_train_final,y_train_final).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = create_dataset(X_val,y_val).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


def process_byte_image(byte_array):
    # Ensure we're working with TensorFlow ops
    img = tf.io.decode_jpeg(byte_array, channels=3)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def preprocess_test_data(inp):
    # def read_hdf5(id_tensor):
    #     # Convert tensor to string
    #     id_str = id_tensor.numpy().decode('utf-8')
    #     # Read from HDF5 and return the raw bytes
    #     return testing_hdf5[id_str][()].tobytes()

    # Wrap the HDF5 reading operation in py_function and specify the output type
    # byte_array = tf.py_function(
    #     read_hdf5,
    #     [isic_id],
    #     Tout=tf.string
    # )
    
    # Process the image using TF operations
    image = process_byte_image(inp['images'])
    inp['images'] = image
    # return ({"image": image, "tabular": tabular_data})
    return (inp)

def load_image_bytes(isic_id):
    with h5py.File('/kaggle/input/isic-2024-challenge/test-image.hdf5', "r") as hdf5_file:
        return hdf5_file[isic_id][()].tobytes()
    
def create_test_dataset(df):
    # Ensure tabular data is float32
    tabular_data = df.drop(columns=['isic_id']).values.astype('float32')
    # Convert ISIC IDs to strings
    isic_ids = df['isic_id'].astype(str).values

    # imgs = [None]*len(isic_ids)
    # for i, isic_id in enumerate(tqdm(isic_ids, desc="Loading testing Images ")):
    #     imgs[i] = testing_hdf5[isic_id][()].tobytes()
     # Use ProcessPoolExecutor for multiprocessing
    imgs = [None] * len(isic_ids)
    with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust workers based on CPU
        future_to_index = {executor.submit(load_image_bytes, isic_id): i for i, isic_id in enumerate(isic_ids)}
        for future in tqdm(as_completed(future_to_index), total=len(isic_ids), desc="Loading testing images"):
            i = future_to_index[future]
            imgs[i] = future.result()

    inp = {"images": imgs, "tabular":tabular_data}
    dataset = tf.data.Dataset.from_tensor_slices(inp)
    dataset = dataset.map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset



# Create the test dataset
test_dataset = create_test_dataset(X_test).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


 for batch in train_dataset.take(1):
    dict1,labels = batch
    images = dict1['images']
    tabular = dict1['tabular']
    print(f"Image batch shape: {images.shape}")
    print(f"Tabular data shape: {tabular.shape}")
    print(f"Labels shape: {labels.shape}")
    
    print(f"\nData types:")
    print(f"Images dtype: {images.dtype}")
    print(f"Tabular dtype: {tabular.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    
    print(f"\nValue ranges:")
    print(f"Images: (min={tf.reduce_min(images):.2f}, max={tf.reduce_max(images):.2f})")
    print(f"Tabular: (min={tf.reduce_min(tabular):.2f}, max={tf.reduce_max(tabular):.2f})")
    print(f"Labels: (min={tf.reduce_min(labels):.2f}, max={tf.reduce_max(labels):.2f})")
print("\nDataset specification:")
print(train_dataset.element_spec)


for dict1 in test_dataset.take(1):
    # dict1 = dict1[0]
    images = dict1['images']
    tabular = dict1['tabular']
    print("Image batch shape:", images.shape)
    print("Tabular data shape:", tabular.shape)
    
    # Print additional details
    print("\nDetailed information:")
    print(f"Image data type: {images.dtype}")
    print(f"Tabular data type: {tabular.dtype}")
    print(f"Image value range: (min={tf.reduce_min(images):.2f}, max={tf.reduce_max(images):.2f})")

# You can also check the dataset spec directly
print("\nDataset specification:")
print(test_dataset.element_spec)


os.environ["KERAS_BACKEND"] = "tensorflow"


for batch in train_dataset.take(1):
    dict1,labels = batch
    
    images = dict1['images']
    tabular = dict1['tabular']
    
    tab_features_len = tabular.shape[1]
    print(tab_features_len)


img_input = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),name='images')
tab_feat = keras.Input(shape=(tab_features_len,),name='tabular')


@register_keras_serializable()
def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        activation='leaky_relu',
        padding="same",
    )
    return conv_layer(x)

def correct_pad(inputs, kernel_size):
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )

def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = keras.activations.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = keras.activations.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if keras.ops.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='leaky_relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=0.1,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / PATCH_SIZE)
    non_overlapping_patches = layers.Reshape((PATCH_SIZE, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


# create_mobilevit

# Initial conv-stem -> MV2 block.
x = conv_block(img_input, filters=16)
x = inverted_residual_block(
    x, expanded_channels=16 * expansion_factor, output_channels=16
)

 # Downsampling with MV2 block.
x = inverted_residual_block(
    x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
)
x = inverted_residual_block(
    x, expanded_channels=24 * expansion_factor, output_channels=24
)
x = inverted_residual_block(
    x, expanded_channels=24 * expansion_factor, output_channels=24
)

# First MV2 -> MobileViT block.
x = inverted_residual_block(
    x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2
)
x = mobilevit_block(x, num_blocks=2, projection_dim=64)

# Second MV2 -> MobileViT block.
x = inverted_residual_block(
    x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2
)
x = mobilevit_block(x, num_blocks=4, projection_dim=80)

# Third MV2 -> MobileViT block.
x = inverted_residual_block(
    x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
)
x = mobilevit_block(x, num_blocks=4, projection_dim=96)
x = conv_block(x, filters=320, kernel_size=1, strides=1)




# Classification head.
x = layers.GlobalAvgPool2D()(x)
ACTIVATION_FUNCTION = 'leaky_relu'
KERNEL_INITIALIZER = keras.initializers.HeNormal()
# x = Dense(2048,activation = ACTIVATION_FUNCTION,kernel_initializer = KERNEL_INITIALIZER)(x)
y = Dense(256,activation = ACTIVATION_FUNCTION,kernel_initializer = KERNEL_INITIALIZER)(tab_feat)
y = Dropout(0.2)(y)
# y = Dense(512,activation = ACTIVATION_FUNCTION,kernel_initializer = KERNEL_INITIALIZER)(y)
# y = Dropout(0.2)(y)

combined = Concatenate()([x, y])
# z = Dense(1024, activation=ACTIVATION_FUNCTION,kernel_initializer = KERNEL_INITIALIZER)(combined)
# z = Dropout(0.2)(z)
z = Dense(512, activation=ACTIVATION_FUNCTION,kernel_initializer = KERNEL_INITIALIZER)(combined)
# z = BatchNormalization()(z)
z = Dropout(0.2)(z)
z = Dense(256, activation=ACTIVATION_FUNCTION,kernel_initializer = KERNEL_INITIALIZER)(z)
z = Dropout(0.2)(z)
# z = BatchNormalization()(z)
# Classify outputs.
logits = layers.Dense(1,activation='sigmoid')(z)


model = keras.Model(inputs=[img_input, tab_feat], outputs=logits,name='MobileViT')


# keras.utils.plot_model(model, "MobileViT.png", show_shapes=True)


model.summary()


# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=5000,
#     decay_rate=0.9)
boundaries = [300,1000]
values = [1e-3,1e-4,1e-5]
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values, name="PiecewiseConstant"
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule,weight_decay=1e-6),
    loss = keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,alpha=0.4,gamma=2.0,label_smoothing = 0.1),
    # loss = keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(curve='PR', name='PR_pAUC'),keras.metrics.AUC(curve='ROC', name='ROC_pAUC')]
)


# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10, 
    restore_best_weights=True,
    start_from_epoch = 5
)

# reduce_lr = ReduceLROnPlateau(
#     monitor='loss', 
#     factor=0.2, 
#     patience=3, 
#     min_lr=1e-6,
#     start_from_epoch = 2 
# )

checkpoint = ModelCheckpoint(
    filepath='best_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)



class_weights = compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
class_weights = dict(enumerate(class_weights))
# class_weights = {0:0.8,1:10.0}
print("Class Weights:", class_weights)


# history = model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=152,
#         callbacks=[early_stopping, checkpoint],
#         verbose=1,
#         # class_weight=class_weights
#     )


# plt.figure(figsize=(12, 8))

# # Plot training and validation loss
# plt.subplot(2, 1, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Plot training and validation AUC
# plt.subplot(2, 1, 2)
# plt.plot(history.history['PR_pAUC'], label='Training PR_AUC')
# plt.plot(history.history['val_PR_pAUC'], label='Validation PR_AUC')
# plt.plot(history.history['ROC_pAUC'], label='Training ROC_AUC')
# plt.plot(history.history['val_ROC_pAUC'], label='Validation ROC_AUC')
# plt.title('AUC')
# plt.xlabel('Epochs')
# plt.ylabel('AUC')
# plt.legend()

# plt.tight_layout()
# plt.show()



# # now load the best model
custom_objects = {}
# best_model = keras.models.load_model('best_model.keras', custom_objects=custom_objects)
best_model = keras.models.load_model('/kaggle/input/mobilevit/keras/v4/1/best_model_mobilevit_v4.keras')


predictions = best_model.predict(test_dataset, verbose=1)
predictions_series = pd.Series(predictions.flatten(), name='Predictions')
# predictions_series = predictions_series.clip(lower=1e-6, upper=0.999999)


sub_df = pd.concat([X_test['isic_id'],predictions_series],axis=1)
sub_df.columns = ['isic_id','target']
sub_df.head()


 sub_df.to_csv('submission.csv', index=False)




