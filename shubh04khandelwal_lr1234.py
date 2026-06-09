import pandas as pd
import numpy as np

# Load the training data (You might have to change the file path based on how you're working)
train_df = pd.read_csv('/kaggle/input/beginners-hypothesis-25/BH25/Training_Data/train.csv')

print(train_df.head())

# For this notebook, we'll only train for 1000 video samples

train_df = train_df[:3000]


import cv2 

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)
    cap.release()
    frames = np.array(frames)  # Shape: (20, 64, 64, 3)
    return frames.flatten()  # Flatten to 1D array
    
video_features = []
for idx, row in train_df.iterrows():
    video_path = f"/kaggle/input/beginners-hypothesis-25/BH25/Training_Data/Train_Videos/{row['video_id']}.mp4"
    video_features.append(process_video(video_path))

video_features = np.array(video_features)  # Shape: (num_samples, 20*64*64*3)


from sklearn.preprocessing import LabelEncoder

# Encode categorical attributes using the Label Encoder
def encode_categorical_columns(df, columns):
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

categorical_columns = ['element', 'motion', 'power']
train_df, label_encoders = encode_categorical_columns(train_df, categorical_columns)


from sklearn.model_selection import train_test_split


y_speed = train_df[['speed']]
y_summary = train_df['video_summary'].apply(lambda x: eval(x))  # Convert string tuples to actual tuples

y_categorical = train_df[categorical_columns]


def encode_categorical_columns(df, columns):
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values.ravel())  # Use ravel() here
        label_encoders[col] = le
    return df, label_encoders



categorical_columns = ['element', 'motion', 'power']  # Example
train_df, label_encoders = encode_categorical_columns(train_df, categorical_columns)

# Split data for training and validation
X_train_speed, X_val_speed, y_train_speed, y_val_speed = train_test_split(video_features, y_speed, test_size=0.2, random_state=42)
X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(video_features, y_categorical, test_size=0.2, random_state=42)
X_train_summary, X_val_summary, y_train_summary, y_val_summary = train_test_split(video_features, y_summary, test_size=0.2, random_state=42)


# Reshape the video features to (num_samples, frames, height, width, channels)
num_frames = 20  # Assuming 20 frames per video
height = 64
width = 64
channels = 3
X_train_speed = X_train_speed.reshape(-1, num_frames, height, width, channels)
X_val_speed = X_val_speed.reshape(-1, num_frames, height, width, channels)
X_train_cat = X_train_cat.reshape(-1, num_frames, height, width, channels)
X_val_cat = X_val_cat.reshape(-1, num_frames, height, width, channels)


from sklearn.linear_model import LinearRegression

import tensorflow as tf

def create_3d_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(num_frames, height, width, channels)),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for speed 
    ])
    return model

# Create and compile models for speed and categorical attributes
model_continuous = create_3d_cnn_model()
model_continuous.compile(optimizer='adam', loss='mse', metrics=['mae'])

models_categorical = {}
for col in categorical_columns:
    model = create_3d_cnn_model()  # You can adjust output layer for categorical predictions
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Change loss function
    models_categorical[col] = model




model_continuous.fit(X_train_speed, y_train_speed, epochs=10, validation_data=(X_val_speed, y_val_speed))

for col in categorical_columns:
    models_categorical[col].fit(X_train_cat, y_train_cat[col], epochs=10, validation_data=(X_val_cat, y_val_cat[col]))




# Training for video_summary, separate models for x and y here

x_values, y_values = zip(*y_train_summary)

model_summary_x = LinearRegression()
model_summary_x.fit(X_train_summary, x_values)

model_summary_y = LinearRegression()
model_summary_y.fit(X_train_summary, y_values)


import os

test_folder = '/kaggle/input/beginners-hypothesis-25/BH25/Testing_Data'

test_features = []
video_ids = []

# The below function finds all the video ids in the test folder and sorts them, 
# and then stores their features in test_features

for video_id in sorted(
    [f for f in os.listdir(test_folder) if f.endswith(".mp4")], key=lambda x: int(x.split('.')[0])):
    video_path = os.path.join(test_folder, video_id)
    video_ids.append(video_id.split('.')[0]) 
    test_features.append(process_video(video_path))

test_features = np.array(test_features)  # Shape: (num_test_samples, 20*64*64*3)
test_features1 = test_features.reshape(-1, num_frames, height, width, channels)  # Reshape test data


y_test_speed = model_continuous.predict(test_features1)
y_test_summary_x = model_summary_x.predict(test_features)
y_test_summary_y = model_summary_y.predict(test_features)


y_categorical = {}

for col in categorical_columns:
    y_categorical[col] = models_categorical[col].predict(test_features1)





# As we are doing regression in this notebook, we need to label the numbers back to what they represent using
# inverse label transform, i.e. 1 will get labelled back as circular for example.

decoded_categorical = {}
for col, predictions in y_categorical.items():
    rounded_predictions = np.rint(predictions).astype(int)
    valid_classes = range(len(label_encoders[col].classes_))
    rounded_predictions = np.clip(rounded_predictions, min(valid_classes), max(valid_classes))
    decoded_categorical[col] = label_encoders[col].inverse_transform(rounded_predictions.ravel())


output_df = pd.DataFrame({
    'video_id': video_ids,                      
    'element': decoded_categorical['element'],  
    'motion': decoded_categorical['motion'],
    'power': decoded_categorical['power'],
    'speed':  [x[0] for x in y_test_speed],               
    'video_summary': list(zip(y_test_summary_x, y_test_summary_y)) 
})


output_df.to_csv("submission.csv", index=False)

# This is the final submission file to be submitted

