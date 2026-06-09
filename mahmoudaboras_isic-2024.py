import numpy as np
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv')

df.head()



missing_values = df.isnull().sum()

print("Missing values before filling:")
df.isnull().sum()


for column in df.columns:
    mode_value = df[column].mode()[0]  
    df[column].fillna(mode_value, inplace=True)


missing_values_after_filling = df.isnull().sum()
df.isnull().sum()


df.head()


# Add the image filename
df['image'] = df['isic_id'].astype(str) + '.jpg'

# Convert target to string labels for categorical classification (optional)
df['label'] = df['target'].astype(str)



import h5py

# Load and inspect the HDF5 file to check available keys
with h5py.File('/kaggle/input/isic-2024-challenge/train-image.hdf5', 'r') as f:
    # List all the keys in the HDF5 file
    print(list(f.keys()))

    # Check the structure of the file
    for key in f.keys():
        print(f"Key: {key}, Type: {type(f[key])}, Shape: {f[key].shape}")



X = f['data'][:]  # or whatever key corresponds to the image data

# Normalize
X = X / 255.0

# Labels
y = df['target'].values
y = to_categorical(y, num_classes=2)  # convert to one-hot encoding


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()



model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)


