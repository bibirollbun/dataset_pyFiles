pip install -U --upgrade tensorflow


from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os


DATA_FOLDER = '/kaggle/input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

print(f"train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")


train_sample_metadata = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_sample_metadata.head()


train_sample_metadata.groupby('label')['label'].count().plot(figsize=(5,5),kind='bar',title='The Label in the Training Set')
plt.show()


train_sample_metadata.shape


f_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.label=='FAKE'].sample(5).index)
f_train_sample_video


def capture_image_from_video(video_path):
    # Capture video
    capture_image = cv2.VideoCapture(video_path)
    ret, frame = capture_image.read()
    
    if not ret:
        print(f"Could not read frame from {video_path}")
        return
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  
    ax.imshow(frame)
    plt.show()
    capture_image.release()


for video_file in f_train_sample_video:
    capture_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))


r_train_sample_video = list(train_sample_metadata.loc[train_sample_metadata.label=='REAL'].sample(5).index)
r_train_sample_video


for video_file in r_train_sample_video:
    capture_image_from_video(os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER,video_file))


f_videos = list(train_sample_metadata.loc[train_sample_metadata.label=='FAKE'].index)
from IPython.display import HTML
from base64 import b64encode

def play_video(video_file,subset=TRAIN_SAMPLE_FOLDER):
    video_url = open(os.path.join(DATA_FOLDER,subset,video_file),'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" %data_url)
play_video(f_videos[5])


img_size = 224
batch_size = 64
epochs = 15

max_seq_length = 20
num_features = 2048


def crop_center_square(frame):
    y,x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y :start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(img_size, img_size)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while 1:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def pretrain_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
    weights = "imagenet",
    include_top=False,
    pooling="avg",
    input_shape = (img_size,img_size,3)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    
    inputs = keras.Input((img_size,img_size,3))
    preprocessed = preprocess_input(inputs)
    
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = pretrain_feature_extractor()


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = list(df.index)
    labels = df["label"].values
    labels = np.array(labels=='FAKE').astype(np.int)
    
    frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, max_seq_length, num_features), dtype="float32" 
    )
    
    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]
        
        temp_frame_mask = np.zeros(shape=(1, max_seq_length,), dtype="bool")
        temp_frame_features = np.zeros(shape=(1, max_seq_length, num_features), dtype="float32")
        
        for i, batch in enumerate(frames):
            video_length = batch.shape[0] 
            length = min(max_seq_length, video_length) #if length is over 20s ,only cut 20s
            for j in range(length):
                temp_frame_features[i, j, :] =feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] =1 # 1 = not masked, 0 = masked ->give 1 when there are images ,otherwise 0 for padding
        
        frame_features[idx,] =temp_frame_features.squeeze() #squeeze array for training
        frame_masks[idx,] =temp_frame_mask.squeeze()
    
    return (frame_features, frame_masks), labels


from sklearn.model_selection import train_test_split

Train_set , Test_set = train_test_split(train_sample_metadata, test_size=0.3,random_state=42,
                                       stratify=train_sample_metadata['label'])
print(Train_set.shape, Test_set.shape)


train_data, train_labels = prepare_all_videos(Train_set, "train")
test_data, test_labels = prepare_all_videos(Test_set, "test")

print(f"Frame features in train set:{train_data[0].shape}")
print(f"Frame masks in train set:{train_data[1].shape}")


frame_features_input = keras.Input((max_seq_length, num_features))
mask_input = keras.Input((max_seq_length,),dtype="bool")

x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask = mask_input)
x = keras.layers.GRU(8)(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model([frame_features_input, mask_input], output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


checkpoint = keras.callbacks.ModelCheckpoint('./', save_weights_only=True, save_best_only=True)
history = model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_data=([test_data[0], test_data[1]], test_labels),
        callbacks=[checkpoint],
        epochs=epochs,
        batch_size=8
)


model.save("/kaggle/working/fake_video_detection_model.h5")


import os
print(os.listdir("/kaggle/working"))


test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, max_seq_length,), dtype="bool")
    frame_features = np.zeros(shape=(1, max_seq_length, num_features), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_seq_length, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path):
    frames = load_video(os.path.join(DATA_FOLDER, TEST_FOLDER,path))
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]
    
# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")


test_video = np.random.choice(test_videos["video"].values.tolist())
print(f"Test video path: {test_video}")

if(sequence_prediction(test_video)>=0.5):
    print(f'The predicted class of the video is FAKE')
else:
    print(f'The predicted class of the video is REAL')

play_video(test_video,TEST_FOLDER)


import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from IPython.display import HTML
import base64

# Load the trained model
model = keras.models.load_model("/kaggle/working/fake_video_detection_model.h5")

# Define constants
max_seq_length = 20  # Should be the same as in training
num_features = 2048   # Adjust according to feature extractor

# Load the feature extractor (same one used in training)
feature_extractor = keras.applications.InceptionV3(include_top=False, pooling="avg", input_shape=(224, 224, 3))

def load_video(video_path, max_frames=max_seq_length, frame_size=(224, 224)):
    """Loads video frames and resizes them to the required shape."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        frame = cv2.resize(frame, frame_size)  # Resize to match feature extractor
        frame = frame / 255.0  # Normalize
        frames.append(frame)
        frame_count += 1

    cap.release()
    
    if len(frames) < max_frames:
        # Pad with zeros if less frames
        frames += [np.zeros((224, 224, 3))] * (max_frames - len(frames))
    
    return np.array(frames)

def preprocess_video(video_path):
    """Preprocesses a video and extracts features for prediction."""
    frames = load_video(video_path)
    frames = frames[None, ...]  # Add batch dimension
    
    frame_mask = np.zeros((1, max_seq_length), dtype="bool")
    frame_features = np.zeros((1, max_seq_length, num_features), dtype="float32")

    video_length = frames.shape[1]
    length = min(max_seq_length, video_length)

    for i in range(length):
        frame_features[0, i, :] = feature_extractor.predict(frames[:, i, :, :, :])
    frame_mask[0, :length] = 1

    return [frame_features, frame_mask]

# Path to the test video
video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/bwdmzwhdnw.mp4"

# Preprocess the video
video_data = preprocess_video(video_path)

# Make prediction
prediction = model.predict(video_data)
predicted_label = "FAKE" if prediction[0][0] > 0.5 else "REAL"

print(f"Prediction: {predicted_label} (Confidence: {prediction[0][0]:.4f})")

# Function to display the video in a Kaggle Notebook
def display_video(video_path):
    """Encodes and displays video in a Kaggle Notebook"""
    video_encoded = base64.b64encode(open(video_path, "rb").read()).decode('utf-8')
    video_tag = f'''
    <video width="600" controls>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
    </video>
    '''
    return HTML(video_tag)

# Display the video
display_video(video_path)

