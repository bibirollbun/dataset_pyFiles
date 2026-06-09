import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df


df.shape


df.isnull().sum()
# No null values are found


df.dtypes


df['Soil Type'].unique()


df['Crop Type'].unique()


df['Fertilizer Name'].unique()


# Label encoding soil type to convert it from object to numeric datatype
from sklearn.preprocessing import LabelEncoder

# Create the encoder
le_soil = LabelEncoder()

# Fit and transform the 'Soil Type' column
df['Soil Type Encoded'] = le_soil.fit_transform(df['Soil Type'])

# Get the mapping of labels
soil_mapping = dict(zip(le_soil.classes_, le_soil.transform(le_soil.classes_)))

# Display the mapping
print("Soil Type Label Mapping:")
for label, code in soil_mapping.items():
    print(f"{label} → {code}")


# Label encoding crop type to convert it from object to numeric datatype
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder for Crop Type
le_crop = LabelEncoder()

# Fit and transform the 'Crop Type' column
df['Crop Type Encoded'] = le_crop.fit_transform(df['Crop Type'])

# Get the mapping from string labels to numeric codes
crop_mapping = dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_)))

# Display the mapping
print("Crop Type Label Mapping:")
for label, code in crop_mapping.items():
    print(f"{label} → {code}")


# Label encoding fertilizer name to convert it from object to numeric datatype
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder for Fertilizer Name
le_fert = LabelEncoder()

# Fit and transform the 'Fertilizer Name' column
df['Fertilizer Name Encoded'] = le_fert.fit_transform(df['Fertilizer Name'])

# Get mapping dictionary
fert_mapping = dict(zip(le_fert.classes_, le_fert.transform(le_fert.classes_)))

# Display the mapping
print("Fertilizer Name Label Mapping:")
for label, code in fert_mapping.items():
    print(f"{label} → {code}")


df.columns


df_cleaned = df.drop(columns=['id', 'Fertilizer Name', 'Soil Type', 'Crop Type'])

# Define X and y
X = df_cleaned.drop(columns=['Fertilizer Name Encoded'])  # Features
y = df_cleaned['Fertilizer Name Encoded']                 # Target


from sklearn.model_selection import train_test_split

# Split into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Number of classes
num_classes = df['Fertilizer Name Encoded'].nunique()

model = Sequential()

# Input layer
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden layer 1
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Hidden layer 2 (optional)
model.add(Dense(32, activation='relu'))

# Output layer
model.add(Dense(num_classes, activation='softmax'))


model.summary()


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Use this for integer labels (not one-hot)
    metrics=['accuracy']
)


history = model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
          epochs=15, 
          batch_size=512, 
          verbose=1)


df2 = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df2


df2['Soil Type Encoded'] = le_soil.transform(df2['Soil Type'])
df2['Crop Type Encoded'] = le_crop.transform(df2['Crop Type'])


# Step 4: Create the final feature set (must match training features)
X_final_test = df2.drop(columns=['id', 'Soil Type', 'Crop Type'])


# Step 5: Predict
y_pred = model.predict(X_final_test)
y_pred_classes = y_pred.argmax(axis=1)  # Get the class with highest probability


# Step 6: Map back from label to fertilizer name (optional)
reverse_fert_mapping = {v: k for k, v in fert_mapping.items()}
predicted_fertilizers = [reverse_fert_mapping[i] for i in y_pred_classes]


# Step 7: Create submission file
submission = pd.DataFrame({
    'id': df2['id'],
    'Fertilizer Name': predicted_fertilizers
})

# Step 8: Save to CSV
submission.to_csv('submission.csv', index=False)




