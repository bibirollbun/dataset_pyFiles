import pandas as pd
import keras
import tensorflow as tf
import h5py

# Load data
train_images = h5py.File('/kaggle/input/isic-2024-challenge/train-image.hdf5', 'r')

IMAGE_SIZE = [128, 128]
AUTO = tf.data.experimental.AUTOTUNE

def decode_images(data):
    # Read jpeg image
    file_bytes = data
    image = tf.io.decode_jpeg(file_bytes)
    
    # Resize
    image = tf.image.resize(image, size=IMAGE_SIZE, method='area')
    
    # Rescale image
    image = tf.cast(image, tf.float32)
    image /= 255.0
    
    # Reshape
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    
    return image

def build_dataset(isic_ids, hdf5):
    images = [None]*len(isic_ids)
    for index, isic_id in enumerate(isic_ids):
        images[index] = hdf5[isic_id][...]
        
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(
      decode_images,
      num_parallel_calls=AUTO
    )

    dataset = dataset.batch(32, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)

    return dataset

# create datasets
train_ids = list(train_images.keys())
train_ds = build_dataset(
    train_ids,
    train_images
)

simple_net = keras.saving.load_model(
    '/kaggle/input/vlevy_isic2024_images_only/keras/default/1/best_model_images_only.keras'
)

simple_net_train = simple_net.predict(train_ds).squeeze()


import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train_df = pd.read_csv(
    '/kaggle/input/isic-2024-challenge/train-metadata.csv', low_memory=False
)
test_df = pd.read_csv(
    '/kaggle/input/isic-2024-challenge/test-metadata.csv', low_memory=False
)
X_train = train_df[test_df.columns.to_list()]
y_train = train_df['target']

numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
X_train_num = X_train[numeric_cols].fillna(train_df[numeric_cols].median())
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)

k = min(20, X_train_num_scaled.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_num_selected = selector.fit_transform(X_train_num_scaled, y_train)
print(X_train_num_selected.shape)
X_train_num_selected = np.c_[X_train_num_selected, simple_net_train.tolist()]
print(X_train_num_selected.shape)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_cat_encoded = encoder.fit_transform(train_df[cat_cols])

X_train_all = sparse.hstack([X_train_num_selected, X_train_cat_encoded]).tocsr()
print(X_train_all.shape)

folds = 6
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

cfs = []
run = 0
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_train)):
    X_tr, y_tr = X_train_all[train_idx], y_train.iloc[train_idx]

    if run % 2 == 0:
        clf = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum()
        )
    else:
        clf = RandomForestClassifier(
            random_state=42,
            max_depth=4
        )

    cfs.append(clf.fit(X_tr, y_tr))


test_images = h5py.File('/kaggle/input/isic-2024-challenge/test-image.hdf5', 'r')
testing_ids = list(test_images.keys())
testing_ds = build_dataset(
    testing_ids,
    test_images
)

simple_net_test = simple_net.predict(testing_ds).squeeze()

X_test_num = test_df[numeric_cols].fillna(train_df[numeric_cols].median())
X_test_num_scaled = scaler.transform(X_test_num)

X_test_num_selected = selector.transform(X_test_num_scaled)
print(X_test_num_selected.shape)
X_test_num_selected = np.c_[X_test_num_selected, simple_net_test.tolist()]
print(X_test_num_selected.shape)

X_test_cat_encoded = encoder.transform(test_df[cat_cols])

X_test_all = sparse.hstack([X_test_num_selected, X_test_cat_encoded]).tocsr()
print(X_test_all.shape)

preds = np.zeros(X_test_all.shape[0])
batch_size=75000
for clf in cfs:
    clf_preds = []
    for i in range(0, X_test_all.shape[0], batch_size):
        batch = X_test_all[i:i+batch_size]
        batch_preds = clf.predict_proba(batch)[:, 1]
        clf_preds.extend(batch_preds)
    preds = np.array(clf_preds) / folds

pred_df = pd.DataFrame({
    'isic_id': testing_ids,
    'target': np.nan_to_num(preds, nan=0.0)
})
pred_df.to_csv('submission.csv', index=False)
pred_df.head()

