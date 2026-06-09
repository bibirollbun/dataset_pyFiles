import os, glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

import pydicom
import imageio.v3 as iio
import skimage

IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH = 128, 128, 32
MIN_WEEK, MAX_WEEK = -12, 133
LSTM_REGULARISER_LAMBDA = 0.05
EPOCHS = 32
SIGMA_MIN, DELTA_MAX = 70, 1000
SQRT_2 = tf.constant(tf.sqrt(2.))

import kagglehub
# kagglehub.login()
# kagglehub.competition_download('osic-pulmonary-fibrosis-progression')
DIR = '../input/osic-pulmonary-fibrosis-progression'
# NPY_DIR = '../input/osic-dataset-npy-format/osic-npy'
NPY_DIR = None
SUBMISSION_DIR = '.'
# MODEL = ''


def copy_dcm_to_local(mode, patient_id, depth=32):
    inp_dir = f'{DIR}/{mode}/{patient_id}'
    out_dir = f'{SUBMISSION_DIR}/{mode}'
    out_file = f'{SUBMISSION_DIR}/{mode}/{patient_id}.npy'
    
    paths = glob.glob(f'{inp_dir}/*.dcm')
    keys = (int(path.split('/')[-1].split('.')[0]) for path in paths)
    paths_keys = sorted(zip(paths, keys), key=lambda x: x[1])
    paths = [elem[0] for elem in paths_keys]

    img_3ds = []
    for idx in range(depth):
        pos = int(idx * len(paths) / depth)
        
        inp_filename = paths[pos]
        os.system(f'mkdir -p {out_dir}')    
        
        # os.system(f'gdcmconv --raw {inp_filename} compressed.dcm 2> nul')
        # os.system(f'dcmj2pnm +on compressed.dcm compressed.png')

        img = pydicom.dcmread(inp_filename).pixel_array
        img_3ds.append(img)

    img_3ds = np.array(img_3ds)
    np.save(out_file, img_3ds)
    return img_3ds

# copy_dcm_to_local('train', 'ID00011637202177653955184', depth=IMG_DEPTH)
    
if NPY_DIR == None:
    NPY_DIR = '.'
    for mode in ('train', 'test'):
        patient_ids = pd.read_csv(f'{DIR}/{mode}.csv')['Patient'].unique()
        for patient_id in tqdm(patient_ids):
            copy_dcm_to_local(mode, patient_id, depth=IMG_DEPTH)


DEVs = tf.config.list_physical_devices()
tf.config.set_visible_devices(DEVs)
print(tf.config.get_visible_devices())


df = pd.read_csv(f'{DIR}/train.csv') \
    .reset_index(drop=True) \
    .groupby(['Patient', 'Weeks']) \
    .agg({
        'FVC': 'mean',
        'Percent': 'mean',
        'Age': 'first',
        'Sex': 'first',
        'SmokingStatus': 'first',
    }) \
    .reset_index()

df


def scans_3d(directory, patient_id, width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    dicom_directory = f'{directory}/{patient_id}'
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_directory)
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_directory, series_IDs[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    img = sitk.GetArrayFromImage(reader.Execute())
    img = skimage.transform.resize(img, (depth, height, width))
    return img

# scans_3d(f'{DIR}/train', 'ID00011637202177653955184').shape


# def path_scans(patient_id):
#     sitk.ProcessObject_SetGlobalWarningDisplay(False)
#     dicom_directory = f'{DIR}/train/{patient_id}'
#     series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_directory)
#     file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_directory, series_IDs[0])
    
#     scan_depth = len(file_names)
#     file_names = file_names[-1]

#     reader = sitk.ImageFileReader()
#     reader.SetFileName(file_names)
#     reader.ReadImageInformation()
#     img_dim = reader.GetSize()[:-1]
#     return img_dim, scan_depth

def patients(df):
    def fvc_agg(g):
        pos = g[g['Weeks'] >= 0].sort_values(by='Weeks', ascending=True)
        neg = g[g['Weeks'] < 0].sort_values(by='Weeks', ascending=False)

        fvc = 0
        if pos.iloc[0]['Weeks'] == 0:
            # has data at week 0
            fvc = pos.iloc[0]['FVC']
        else:
            if neg.shape[0] > 0:
                # has data before and after week 0
                (x1, y1) = neg.iloc[0][['Weeks', 'FVC']]
                (x2, y2) = pos.iloc[0][['Weeks', 'FVC']]

                # Linear interpolation
                fvc = (x2 * y1 - x1 * y2) / (x2 - x1)
            else:
                # only has data after week 0
                (x1, y1) = pos.iloc[0][['Weeks', 'FVC']]
                fvc = y1

        fvc_full = pos.iloc[0]['FVC'] / pos.iloc[0]['Percent'] * 100
        return pd.Series({'FVC_0': fvc, 'FVC_full': fvc_full, 'Ratio_0': fvc / fvc_full})

    df_fvc_0 = df.groupby(['Patient'])[['Weeks', 'FVC', 'Percent']].apply(fvc_agg).reset_index()
    df_patients = df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates().reset_index(drop=True)
    # df_patients[['ScanDim', 'ScanDepth']] = df_patients['Patient'].apply(lambda _id: path_scans(_id)).apply(pd.Series)
    return df_patients.merge(df_fvc_0, on='Patient').set_index('Patient')

df_patients = patients(df)
df_patients


# def eda_scans():
#     fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
#     df_patients.value_counts('ScanDim').plot.barh(ax=axes[0])
#     df_patients['ScanDepth'].hist(ax=axes[1], bins=25)

#     avg_depth = df_patients['ScanDepth'].mean()
#     axes[1].axvline(avg_depth, color='r')

#     axes[0].set_title('Image dimensions (W x H)')
#     axes[1].set_title('Image scan depths (D)')
#     plt.show()

# eda_scans()
# df_patients['ScanDepth'].mean()


def eda():
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    ages = df_patients['Age']
    ages_m = df_patients.loc[df_patients['Sex'] ==   'Male', 'Age']
    ages_f = df_patients.loc[df_patients['Sex'] == 'Female', 'Age']
    sexes = df_patients['Sex'].value_counts()
    statuses = df_patients['SmokingStatus'].value_counts()

    axes[0].hist(x=ages)
    axes[1].hist(x=[ages_m, ages_f], label=['Male', 'Female'])
    axes[1].legend()
    axes[2].bar(x=sexes.index, height=sexes.values, color=['tab:blue', 'tab:orange'])
    axes[3].bar(x=statuses.index, height=statuses.values)

    axes[0].set_title('Distribution of age')
    axes[1].set_title('Distribution of age for each gender')
    axes[2].set_title('Distribution of gender')
    axes[3].set_title('Distribution of SmokingStatus')
    plt.show()

eda()


def progression(status, color, ax=None):
    patients_with_status = df_patients.loc[df_patients['SmokingStatus'] == status] \
        .sample(10, replace=True).index

    for patient_id in patients_with_status:
        df_patient = df.loc[df['Patient'] == patient_id]

        weeks, fvcs = df_patient[['Weeks', 'Percent']].T.values
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Percent')
        ax.plot(weeks, fvcs, color=color)
        ax.tick_params(axis='y', labelcolor=color)

fig, ax1 = plt.subplots()
progression('Ex-smoker', color='tab:orange', ax=ax1)
progression('Never smoked', color='tab:blue', ax=ax1)
progression('Currently smokes', color='tab:red', ax=ax1)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


df_patients


df_fvc_ratios = df.pivot(index='Patient', columns=['Weeks'], values=['Percent']) \
    .droplevel(0, axis=1) \
    .reindex(columns=range(MIN_WEEK, MAX_WEEK + 1)) \
    .interpolate(axis='columns') \
    .bfill(axis=1) \
    .map(lambda x: x / 100) \
    .merge(df_patients[['FVC_full']], left_index=True, right_index=True)

df_fvc_ratios


class OSICTimeSeriesDataset(tf.keras.utils.Sequence):
    def __init__(self, mode, DIR, csv_file,
                 depth=64,
                 batch_size=1,
                 input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH),
                 shuffle=True):

        self.mode = mode
        self.directory = DIR
        self.df = pd.read_csv(csv_file) \
            .reset_index(drop=True) \
            .groupby(['Patient', 'Weeks']) \
            .agg({
                'FVC': 'mean',
                'Percent': 'mean',
                'Age': 'first',
                'Sex': 'first',
                'SmokingStatus': 'first',
            }) \
            .reset_index()

        self.df_patients = self._patients()
        self.df_fvc_ratios = self._fvc_ratios()

        self.X_patients = self.df_patients.copy()
        self.X_patients['Sex'] = self.X_patients['Sex'].map({'Male': 0., 'Female': 1.})
        self.X_patients['SmokingStatus'] = self.X_patients['SmokingStatus'].map({
            'Currently smokes': 0., 'Ex-smoker': 0.5, 'Never smoked': 1.
        })

        # self.X_patients = self.X_patients.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        self.X_patients = self.X_patients.to_numpy()

        self.depth = depth
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df_patients)

    def _patients(self):
        def __fvc_full(g):
            fvc_full = g.iloc[0]['FVC'] / g.iloc[0]['Percent'] * 100
            return pd.Series({'FVC_full': fvc_full})

        df_fvc_full = self.df.groupby(['Patient'])[['Weeks', 'FVC', 'Percent']] \
            .apply(__fvc_full).reset_index()
        df_patients = self.df[['Patient', 'Age', 'Sex', 'SmokingStatus']] \
            .drop_duplicates().reset_index(drop=True)

        df_patients = df_patients.merge(df_fvc_full, on='Patient').set_index('Patient')
        return df_patients

    def _fvc_ratios(self):
        return self.df.pivot(index='Patient', columns=['Weeks'], values=['Percent']) \
            .droplevel(0, axis=1) \
            .reindex(columns=range(MIN_WEEK, MAX_WEEK + 1)) \
            .interpolate(axis='columns') \
            .bfill(axis=1) \
            .map(lambda x: x / 100)

    def _scans_3d(self, patient_id):
        filename = f'{self.directory}/{patient_id}.npy'
        width, height, depth = self.input_size
        
        img_3d = np.load(filename)
        img_3d = skimage.transform.resize(img_3d, (depth, height, width))
        img_3d = np.transpose(img_3d, (1, 2, 0))
        return img_3d
    
    def on_epoch_end(self):
        pass

    def get_ids_batch(self, idx):
        sta = idx * self.batch_size
        fin = min(sta + self.batch_size, len(self.df_patients))
        return self.df_patients[sta:fin].index.to_list()

    def __getitem__(self, idx):
        sta = idx * self.batch_size
        fin = min(sta + self.batch_size, len(self.df_patients))
        df_patients = self.df_patients.iloc[sta:fin]
        X_patients = self.X_patients[sta:fin]
        X_fvc_full = df_patients['FVC_full'].to_numpy()

        imgs_3d = []
        for patient_id in df_patients.index:
            imgs_3d.append(self._scans_3d(patient_id))

        imgs_3d = np.array(imgs_3d)
        
        if self.mode == 'train':
            df_fvc_ratios = self.df_fvc_ratios.loc[df_patients.index.tolist()]
            y = np.concatenate([
                df_fvc_ratios.to_numpy(),
                np.expand_dims(X_fvc_full, axis=1)
            ], axis=1)
            return (imgs_3d, X_patients, X_fvc_full), y
        else:
            return (imgs_3d, X_patients, X_fvc_full)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

osic_time_series_dataset = OSICTimeSeriesDataset('train', f'{NPY_DIR}/train', f'{DIR}/train.csv', batch_size=16)


X, _ = osic_time_series_dataset[2]
fig, ax = plt.subplots(ncols=8, figsize=(24, 3))
for idx in range(0, 16, 2):
    ax.flat[idx // 2].imshow(X[0][0][:, :, idx], cmap='gray')
    ax.flat[idx // 2].axis('off')
plt.show()


# https://keras.io/examples/vision/3D_image_classification/#define-a-3d-convolutional-neural-network

def cnn_3d_regression_features(width, height, depth, dense_units=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    feature_vector = layers.Dense(units=dense_units, activation="relu", name="feature_vector")(x)

    # Define the model.
    model = keras.Model(inputs, feature_vector, name="cnn_3d_regression_features")
    return model

cnn_3d_model = cnn_3d_regression_features(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH)
# cnn_3d_model = keras.saving.load_model('./weights/251010-cnn-3d.keras')
cnn_3d_model.summary()


def full_model(width, height, depth, activation=['linear', 'linear'], hidden_units=64, dense_units=112):
    """Build a 3D convolutional neural network model."""

    # cnn_3d_model = keras.saving.load_model('../input/cnn-3d/keras/default/1/251010-cnn-3d.keras')

    cnn_3d_model = cnn_3d_regression_features(width, height, depth)
    
    img_inputs = cnn_3d_model.input
    feature_inputs = keras.Input((4,))
    fvc_full = keras.Input((1,))
    
    feature_cnn = cnn_3d_model.outputs[0]
    x = layers.Concatenate()([feature_inputs, feature_cnn])
    
    x = layers.Reshape((1, 68))(x)
    x = layers.LSTM(
        hidden_units, activation=activation[0],
        kernel_regularizer=keras.regularizers.l2(LSTM_REGULARISER_LAMBDA), 
        recurrent_regularizer=keras.regularizers.l2(LSTM_REGULARISER_LAMBDA), 
        bias_regularizer=keras.regularizers.l2(LSTM_REGULARISER_LAMBDA)
    )(x)
    outputs = layers.Dense(units=dense_units, activation='sigmoid', name='output')(x)
    outputs = layers.Concatenate()([outputs, fvc_full])
    
    # Define the model.
    model = keras.Model([img_inputs, feature_inputs, fvc_full], outputs, name="full_model")
    return model

model = full_model(width=IMG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH, dense_units=MAX_WEEK-MIN_WEEK+1)
# model = keras.models.load_model(MODEL)
model.summary()


@keras.saving.register_keras_serializable()
def competition_metric(res_true, res_pred):
    """OSIC Competition Metric"""
    fvc = res_true[:, -1:]
    y_true = res_true[:, :-1]
    y_pred = res_pred[:, :-1]
    
    fvc = tf.cast(fvc, tf.float32)
    y_true = tf.cast(y_true, tf.float32) * fvc
    y_pred = tf.cast(y_pred, tf.float32) * fvc

    sigma = tf.constant(100.)
    sigma_clipped = tf.maximum(sigma, SIGMA_MIN)

    delta = tf.abs(y_true - y_pred)
    delta_clipped = tf.minimum(delta, DELTA_MAX)

    # print('y_true = ', y_true[0, 0:100:15])
    # print('y_pred = ', y_pred[0, 0:100:15])

    metric = delta_clipped * SQRT_2 / sigma_clipped + tf.math.log(sigma_clipped * SQRT_2)
    return tf.keras.backend.mean(metric)

@keras.saving.register_keras_serializable()
def modified_mae_loss(res_true, res_pred):
    fvc = res_true[:, -1:]
    y_true = res_true[:, :-1]
    y_pred = res_pred[:, :-1]
    
    y_true = tf.cast(y_true, tf.float32) * fvc
    y_pred = tf.cast(y_pred, tf.float32) * fvc

    return tf.reduce_mean(tf.abs(y_true - y_pred))


tf.keras.backend.clear_session()
with tf.device('/GPU:0'):
    model.compile(optimizer='adam', loss=modified_mae_loss, metrics=[competition_metric], run_eagerly=True)
    history = model.fit(osic_time_series_dataset, epochs=EPOCHS)
    model.save('251119-cnn-rnn.keras')


plt.plot(history.history['loss'])
plt.title('Model Loss per epochs')
plt.xticks(range(len(history.history['loss'])))
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.savefig('loss_per_epochs.png')
plt.show()


x, y = osic_time_series_dataset[0]
y_pred = model.predict(x)

xs = range(MIN_WEEK, MAX_WEEK + 1)
plt.plot(xs, y_pred[0][:-1] * y_pred[0][-1])
plt.plot(xs, y[0][:-1] * y_pred[0][-1])
# plt.xlim(MIN_WEEK, MAX_WEEK)
plt.xticks(range(MIN_WEEK, MAX_WEEK + 1, 10))
plt.show()


osic_time_series_dataset_test = OSICTimeSeriesDataset('test', f'{NPY_DIR}/test', f'{DIR}/test.csv', batch_size=3)
osic_time_series_dataset_test


df_y_test_pred = []

for idx in range(len(osic_time_series_dataset_test)):
    imgs, patients, fvc = osic_time_series_dataset_test[idx]
    with tf.device('/GPU:0'):
        y_test_pred = model.predict([imgs, patients, fvc])
        y_test_pred = y_test_pred[:, :-1] * y_test_pred[:, -1:]

    df_y_test_pred_ = pd.DataFrame(y_test_pred)
    df_y_test_pred_.index = osic_time_series_dataset_test.get_ids_batch(idx)
    df_y_test_pred_.columns += MIN_WEEK
    df_y_test_pred.append(df_y_test_pred_)

df_y_test_pred = pd.concat(df_y_test_pred)
df_y_test_pred

df_y_test_pred = df_y_test_pred.reset_index(names='Patient') \
    .melt(id_vars=['Patient'], var_name='Week', value_name='FVC')

df_y_test_pred['Confidence'] = 100
df_y_test_pred['Patient_Week'] = df_y_test_pred['Patient'] + '_' + df_y_test_pred['Week'].astype('str')
df_y_test_pred = df_y_test_pred.set_index('Patient_Week')
df_y_test_pred


df_y_test_pred[['FVC', 'Confidence']] \
    .to_csv(f'{SUBMISSION_DIR}/submission.csv')




