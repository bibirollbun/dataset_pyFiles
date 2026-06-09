!pip install --no-index /kaggle/input/utilss/pylibjpeg-2.0.1-py3-none-any.whl --no-deps
!pip install --no-index /kaggle/input/utilss/pylibjpeg_libjpeg-2.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
!pip install --no-index /kaggle/input/utilss/pylibjpeg_openjpeg-2.4.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps

!pip install --no-index /kaggle/input/utilss/pydicom-3.0.1-py3-none-any.whl --no-deps
!pip install --no-index /kaggle/input/utils-numpy/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
!pip install --no-index /kaggle/input/utils-numpy/scikit_image-0.25.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
train_df


print(train_df.info())


# Class distribution
print(train_df['cancer'].value_counts())



# Check for missing values
print(train_df.isnull().sum())



train_df["age"] = train_df["age"].fillna(train_df["age"].median())
#train_df["BIRADS"] = train_df["BIRADS"].fillna(train_df["BIRADS"].median())
#train_df = train_df.drop(columns=["BIRADS"])
#train_df = train_df.drop(columns=["density"])
train_df = train_df.drop(columns=["machine_id"])
train_df


import os

# Define the base directory where images are stored
BASE_DIR = "/kaggle/input/rsna-breast-cancer-detection/train_images"

# Function to construct full image path
def get_image_path(row):
    return os.path.join(BASE_DIR, str(row["patient_id"]), f"{row['image_id']}.dcm")

# Apply the function to each row
train_df["image_path"] = train_df.apply(get_image_path, axis=1)

# Preview the updated DataFrame
print(train_df)



image_base_path = "/kaggle/input/rsna-breast-cancer-detection/train_images"

train_df['image_path'] = train_df.apply(
    lambda row: f"{image_base_path}/{row['patient_id']}/{row['image_id']}.dcm", 
    axis=1
)


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

df = pd.get_dummies(train_df, columns=['laterality', 'view'], drop_first=False)

# Normalize numerical features
numerical_cols = ['age']  # add more if needed
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

tabular_data = df.drop(columns=['site_id','invasive', 'biopsy', 'image_id','difficult_negative_case', 'patient_id', 'image_path', 'cancer', 'BIRADS', 'density'])
tabular_data = tabular_data.astype(np.float32).values  # ðŸ‘ˆ this is key!
labels = df['cancer'].values
labels = labels.astype(np.float32)  # Binary classification


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

df = pd.get_dummies(train_df, columns=['laterality', 'view'], drop_first=False)

# Normalize numerical features
numerical_cols = ['age']  # add more if needed
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Include all positive samples
positive_df = df[df['cancer'] == 1]

# Sample equal or smaller number of negative examples (or a 1:3 ratio)
negative_df = df[df['cancer'] == 0].sample(n=len(positive_df) * 3, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([positive_df, negative_df]).sample(frac=1, random_state=42).reset_index(drop=True)

tabular_data = df_balanced.drop(columns=['site_id','invasive', 'biopsy', 'image_id','difficult_negative_case', 'patient_id', 'image_path', 'cancer', 'BIRADS', 'density'])
tabular_data = tabular_data.astype(np.float32).values  # ðŸ‘ˆ this is key!
labels = df_balanced['cancer'].astype(np.float32).values
image_paths = df_balanced['image_path'].values.astype(str)



len(tabular_data)


# import pydicom
# import tensorflow as tf
# from tensorflow.keras import layers, models, applications, Input
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from skimage.transform import resize

# # Function to build the multimodal model
# def create_multimodal_model(image_shape=(224, 224, 1), tabular_input_dim=4):
#     # IMAGE INPUT BRANCH
#     image_input = layers.Input(shape=image_shape, name="image_input")

#     # Convert grayscale image to RGB by using a Conv2D layer with 3 filters
#     x = layers.Conv2D(3, (1, 1), padding="same")(image_input)  # Convert grayscale (1 channel) to 3 channels

#     # Use EfficientNetB0 without pre-trained weights
#     base_model = applications.EfficientNetB0(
#         include_top=False,
#         weights=None,  # No pre-trained weights
#         input_tensor=x,  # Use the modified input tensor
#         pooling="avg"  # Global average pooling
#     )

#     # Freeze the pre-trained layers (although there are none since we're training from scratch)
#     base_model.trainable = False

#     # Add custom layers on top of EfficientNetB0
#     x = base_model.output
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)

#     # TABULAR INPUT BRANCH
#     tabular_input = layers.Input(shape=(tabular_input_dim,), name="tabular_input")
#     tabular_x = layers.Dense(64, activation="relu")(tabular_input)
#     tabular_x = layers.Dropout(0.5)(tabular_x)

#     # Concatenate image and tabular branches
#     combined = layers.concatenate([x, tabular_x])

#     # Output layer
#     output = layers.Dense(1, activation="sigmoid")(combined)

#     # Create the model
#     model = models.Model(inputs=[image_input, tabular_input], outputs=output)

#     return model


import pandas as pd
import numpy as np
import tensorflow as tf
import pydicom
from skimage.transform import resize

# ---------- 1. LOAD TEST METADATA ----------
test_df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")

# ---------- 2. CREATE IMAGE PATHS ----------
test_df["image_path"] = test_df.apply(
    lambda row: f"/kaggle/input/rsna-breast-cancer-detection/test_images/{row['patient_id']}/{row['image_id']}.dcm",
    axis=1
)

# ---------- 3. ONE-HOT ENCODE CATEGORICALS ----------
test_df = pd.get_dummies(test_df, columns=["view", "laterality"], drop_first=False)

# ---------- 4. ENSURE EXPECTED FEATURES ----------
expected_features = [
    "age", "implant",
    "view_CC", "view_MLO", "view_AT", "view_LM", "view_LMO", "view_ML",
    "laterality_L", "laterality_R"
]

# Add missing one-hot columns with 0
for col in expected_features:
    if col not in test_df.columns:
        test_df[col] = 0

# ---------- 5. PREPARE INPUTS ----------
tabular_features = expected_features  # Already complete
X_tab_test = test_df[tabular_features].astype(np.float32).values
X_img_test_paths = test_df["image_path"].values
prediction_ids = test_df["prediction_id"].values

NUM_TABULAR_FEATURES = X_tab_test.shape[1]

# ---------- 6. LOAD DICOM ----------
def load_dicom_image(path):
    if isinstance(path, tf.Tensor):
        path = path.numpy().decode("utf-8")
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img = resize(img, (224, 224), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=-1)
    img /= 255.0
    return img

def preprocess_test(image_path, tabular_row):
    image = tf.py_function(load_dicom_image, [image_path], tf.float32)
    image.set_shape([224, 224, 1])
    tabular_row.set_shape([NUM_TABULAR_FEATURES])
    return (image, tabular_row)  # <-- This is good


def create_test_dataset(image_paths, tabular_data, batch_size=32):
    image_paths = tf.constant(image_paths)
    tabular_data = tf.convert_to_tensor(tabular_data, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, tabular_data))
    ds = ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------- 8. CREATE TEST DATASET ----------
test_ds = create_test_dataset(X_img_test_paths, X_tab_test)

# Debug one batch
for images, tabular in test_ds.take(1):
    print("Image batch shape:", images.shape)   # (batch_size, 224, 224, 1)
    print("Tabular batch shape:", tabular.shape) # (batch_size, 13)

# ---------- 9. LOAD TRAINED MODEL ----------
model = tf.keras.models.load_model("/kaggle/input/best-model-1/best_model (5).keras")

# ---------- 10. PREDICT ----------
test_ds_for_pred = test_ds.map(lambda img, tab: ((img, tab),))  # comma ensures it's a tuple of inputs
preds = model.predict(test_ds_for_pred, verbose=1).squeeze()


# ---------- 11. AGGREGATE ----------
submission_df = pd.DataFrame({
    "prediction_id": prediction_ids,
    "cancer": preds
})
submission_df = submission_df.groupby("prediction_id", as_index=False).mean()

# ---------- 12. SAVE ----------
submission_df.to_csv("submission.csv", index=False)
print("submission.csv is saved.")





