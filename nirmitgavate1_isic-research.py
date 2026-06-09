import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import glob
import sys,os
import h5py
from PIL import Image
import io
import PIL
from sklearn.utils import class_weight
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


import pickle

import tensorflow as tf
import keras 
from tensorflow.keras.optimizers import Adam,SGD,AdamW
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Input,GlobalMaxPooling2D,Flatten,Conv2D,Dropout,GlobalAveragePooling2D,BatchNormalization,MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50 ,preprocess_input
# from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input as efnet_pp
from tensorflow.keras.losses import binary_crossentropy 
from tensorflow.keras.regularizers import l2,L2
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.utils import shuffle
import keras_cv
# tensorflow.keras.regularizers


class Config:
    # Image paths
    train_image_path: str = "/kaggle/input/isic-2024-challenge/train-image.hdf5"
    test_image_path: str = "/kaggle/input/isic-2024-challenge/test-image.hdf5"

    # Metadata paths
    train_metadata_path: str = "/kaggle/input/isic-2024-challenge/train-metadata.csv"
    test_metadata_path: str = "/kaggle/input/isic-2024-challenge/test-metadata.csv"

    # Other constants
    image_size: tuple = (128,128)
    batch_size: int = 32
    num_classes: int = 9
    seed: int = 42
    pos_frac: int=5.0
    neg_frac: int=0.01


    label_dict:dict ={0: 'benign', 1: 'malignant'}


#dataframes
train_df=pd.read_csv(Config.train_metadata_path)
test_df=pd.read_csv(Config.test_metadata_path)

#image data
train_images=h5py.File(Config.train_image_path, 'r')
test_images=h5py.File(Config.test_image_path, 'r')


with h5py.File(Config.train_image_path, 'r') as f:
    raw_bytes = f['ISIC_0052109'][()]  

image = Image.open(io.BytesIO(raw_bytes))

title=train_df.query('isic_id=="ISIC_0052109"')['target'].values[0]
plt.imshow(image)
plt.title(Config.label_dict[title])
plt.axis('off')
plt.show()


fig,ax=plt.subplots(1,3,figsize=(12,6))
with h5py.File(Config.test_image_path, 'r') as f:
    test_image_isic_id=[]
    test_bytes=[]
    for val in f.keys():
        test_image_isic_id.append(val)
        test_bytes.append(f[val][()])
    
for i,img in enumerate(test_bytes):
    image=Image.open(io.BytesIO(img))
    ax[i].imshow(image)
    ax[i].axis(False)
plt.suptitle("")
plt.tight_layout()
plt.show()


train_df.head()


test_df.head()


train_df.info()


test_df.info()


train_null_percentage = (train_df.isnull().sum() / len(train_df)) * 100
train_null_percentage[train_null_percentage > 0].sort_values(ascending=False).plot(kind='bar', title='Train Null %')
plt.show()


def remove_columns(train,test):

    train_null_percentage = (train.isnull().sum() / len(train)) * 100
    train_null_percentage[train_null_percentage > 0].sort_values(ascending=False)
    
    train_columns=set(train.columns)
    test_columns=set(test.columns)
    
    drop_cols=list(train_columns-test_columns)
    if "target" in drop_cols:
        drop_cols.remove("target")
    
    drop_cols_lst=list(train_null_percentage[train_null_percentage>90].index)
    drop_cols_lst=drop_cols+drop_cols_lst
    
    train=train.drop(drop_cols_lst,axis=1)
    return train,test
    
train_df,test_df=remove_columns(train_df,test_df)


train_df=train_df.dropna()


train_df['target'].value_counts()


numeric_cols=train_df.select_dtypes(exclude='object').columns
object_cols=train_df.select_dtypes(include='object').columns


def lesion_detector(patient_id):
    subset = train_df[train_df['patient_id']==patient_id]
    
    sns.scatterplot(data=subset, x='tbp_lv_x', y='tbp_lv_y',hue='target',palette='viridis')
    plt.title(f"tbp_lv_x vs tbp_lv_y for patient {patient_id}")
    plt.xlabel("tbp_lv_x")
    plt.ylabel("tbp_lv_y")
    plt.show()
lesion_detector('IP_9577633')


train_df['target'].value_counts()


neg_samples=train_df[train_df['target']==0].sample(frac=Config.neg_frac,random_state=Config.seed)
pos_samples=train_df[train_df['target']==1].sample(frac=Config.pos_frac,random_state=Config.seed,replace=True)


train_df=pd.concat([neg_samples,pos_samples])


train_df['target'].value_counts()


class_weights = compute_class_weight('balanced', classes=np.unique(train_df['target']), y=train_df['target'])
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
class_weights_dict={i:class_weights[i] for i in range(len(class_weights))}


train_df = train_df.reset_index(drop=True)
train_df["fold"] = -1
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Config.seed)
for i, (training_idx, validation_idx) in enumerate(sgkf.split(train_df, y=train_df.target, groups=train_df.patient_id)):
    train_df.loc[validation_idx, "fold"] = int(i)

# Use first fold for training and validation
training_df = train_df.query("fold!=0")
validation_df = train_df.query("fold==0")
print(f"# Num Train: {len(training_df)} | Num Valid: {len(validation_df)}")


cols=['sex','anatom_site_general','tbp_lv_location_simple']
fig,ax=plt.subplots(1,3,figsize=(13,6))
for i,col in enumerate(cols):
    order=train_df[col].value_counts().index
    sns.countplot(data=train_df, x=col, ax=ax[i],order=order)
    ax[i].set_title(f'Distribution of {col}',fontsize=8)
    ax[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


cols=['tbp_lv_location_simple','tbp_lv_location','copyright_license']
fig,ax=plt.subplots(1,3,figsize=(13,6))
# ax=ax.flatten()
for i,col in enumerate(cols):
    order=train_df[col].value_counts().index
    sns.countplot(data=train_df, x=col, ax=ax[i],order=order)
    ax[i].set_title(f'Distribution of {col}',fontsize=8)
    ax[i].tick_params(axis='x', rotation=90)

plt.subplots_adjust(wspace=0.3, hspace=0.9)
plt.tight_layout()


def feature_engineering(df):
    numeric_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
        'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
        'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
        'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
        'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
    ]

    log_features = ['clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_perimeterMM']
    for col in log_features:
        df[col + '_log'] = np.log1p(df[col])
    df['diff_A'] = df['tbp_lv_Aext'] - df['tbp_lv_A']
    df['diff_B'] = df['tbp_lv_Bext'] - df['tbp_lv_B']
    df['diff_C'] = df['tbp_lv_Cext'] - df['tbp_lv_C']
    df['diff_L'] = df['tbp_lv_Lext'] - df['tbp_lv_L']
    df['diff_stdL'] = df['tbp_lv_stdLExt'] - df['tbp_lv_stdL']
    df['diff_deltaLB_deltaL'] = df['tbp_lv_deltaLB'] - df['tbp_lv_deltaL']

    df['minorAxis_times_eccentricity'] = df['tbp_lv_minorAxisMM'] * df['tbp_lv_eccentricity']
    df['deltaL_times_stdL'] = df['tbp_lv_deltaL'] * df['tbp_lv_stdL']
    df['x_times_y'] = df['tbp_lv_x'] * df['tbp_lv_y']
    df['age_group'] = pd.cut(df['age_approx'], bins=[0,20,40,60,80,100], labels=False)
    df['size_category'] = pd.cut(df['clin_size_long_diam_mm'], bins=[-1,5,15,50,200], labels=['small','medium','large','very_large'])
    df['high_color_variance'] = (df['tbp_lv_color_std_mean'] > 0.5).astype(int)
    df['high_symmetry'] = (df['tbp_lv_symm_2axis'] > 0.8).astype(int)
    df['symmetry_angle_bucket'] = pd.cut(df['tbp_lv_symm_2axis_angle'], bins=[0,45,90,135,180], labels=False)
    df['distance_from_center'] = np.sqrt(df['tbp_lv_x']**2 + df['tbp_lv_y']**2)
    df['high_confidence_nevi'] = (df['tbp_lv_nevi_confidence'] > 0.7).astype(int)

    df['area_perimeter_ratio'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-5)
    df['border_color_ratio'] = df['tbp_lv_norm_border'] / (df['tbp_lv_norm_color'] + 1e-5)
    df['color_std_delta'] = df['tbp_lv_color_std_mean'] - df['tbp_lv_radial_color_std_max']
    df['z_score_location'] = (df['tbp_lv_z'] - df['tbp_lv_z'].mean()) / (df['tbp_lv_z'].std() + 1e-5)
    df['eccentricity_times_perimeter'] = df['tbp_lv_eccentricity'] * df['tbp_lv_perimeterMM']

    return df
for df in [train_df,test_df]:
       df=feature_engineering(df)


scaler=StandardScaler()
oe=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ohe = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
X=train_df.drop("target",axis=1)
y=train_df['target'] 


numerical = X.select_dtypes(include='number').columns.tolist()
categorical = X.select_dtypes(include='object').columns.tolist()


ct = ColumnTransformer(
    [
    ("num_preprocess", scaler, numerical),
    ("cat_preprocess", oe, categorical)
    ]
)
X_trans=ct.fit_transform(X)


# X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3, random_state=Config.seed)


# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_dict={}
# for i,wt in enumerate(class_weights):
#     class_weights_dict[i]=wt


training_isic_ids=list(training_df['isic_id'].values)
validation_isic_ids=list(validation_df['isic_id'].values)


train_images_array=[]
validation_images_array=[]
with h5py.File(Config.train_image_path, 'r') as f:
    for id in training_isic_ids:
        raw_bytes=f[id][()]
        img=Image.open(io.BytesIO(raw_bytes)).resize(Config.image_size)
        img_arr=np.asarray(img)
        train_images_array.append(img_arr)
    for id in validation_isic_ids:
        raw_bytes=f[id][()]
        img=Image.open(io.BytesIO(raw_bytes)).resize(Config.image_size)
        img_arr=np.asarray(img)
        validation_images_array.append(img_arr)
    

test_images_array=[]
with h5py.File(Config.test_image_path, 'r') as f:
    for val in f.keys():
        raw_bytes=f[val][()]
        img=Image.open(io.BytesIO(raw_bytes)).resize(Config.image_size)
        img_arr=np.asarray(img)
        test_images_array.append(img_arr)



train_images_array=np.array(train_images_array)
validation_images_array=np.array(validation_images_array)


train_X=train_images_array
valid_X=validation_images_array
train_y=training_df['target'].values
valid_y=validation_df['target'].values
test_X=np.array(test_images_array)


train_X, train_y = shuffle(train_X, train_y, random_state=42)
valid_X, valid_y = shuffle(valid_X, valid_y, random_state=42)


def preprocess(img, label):
    img = tf.image.resize(img, (128,128)) 
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_X, valid_y))
valid_ds = valid_ds.map(lambda img, label: (tf.cast(img, tf.float32)/255.0, label)).batch(32).prefetch(tf.data.AUTOTUNE)


img,label=next(iter(valid_ds))
print(img.shape,label.shape)


image_input = keras.Input(shape=(128,128, 3), name="images")

backbone = keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
x = backbone(image_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)  # Reduced dropout

# Branch for tabular/feature input
x = keras.layers.Dense(96, activation="selu", kernel_regularizer=l2(0.01))(x)
x = keras.layers.Dense(128, activation="selu", kernel_regularizer=l2(0.01))(x)
x = keras.layers.Dropout(0.2)(x) 
x = keras.layers.Dense(128, activation="selu", kernel_regularizer=l2(0.01))(x)
x = keras.layers.Dropout(0.2)(x) 
x = keras.layers.Dense(128, activation="selu", kernel_regularizer=l2(0.01))(x)
x = keras.layers.Dropout(0.2)(x)

out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)


model = keras.models.Model(image_input, out)


model.summary()


def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.where(tf.equal(y_true, 1), 
                      tf.constant(class_weights_dict[1], dtype=tf.float32), 
                      tf.constant(class_weights_dict[0], dtype=tf.float32))
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)


model.compile(
    loss= 'binary_crossentropy',
    metrics=['accuracy'],
    optimizer='Adam'
)



lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


r=model.fit(train_ds,validation_data=valid_ds,epochs=30)


plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()


plt.plot(r.history['accuracy'],label='accuracy')
plt.plot(r.history['val_accuracy'],label='val_accuracy')
plt.legend()


model.predict(test_X)


model.save('base_model.h5')




