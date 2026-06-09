from tensorflow.keras.models import load_model
import os
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import numpy as np
tqdm.pandas()
from pathlib import Path


model = load_model('/kaggle/input/bird_classification_sk/keras/default/1/bird_classification_nw.h5')

IDX_TO_LABEL = sorted(pd.read_csv('/kaggle/input/birdclef-2025/train.csv').primary_label.unique())

df_taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')


model.summary()


# SAMPLING_RATE = 32000

# test_audio_dir = '../input/birdclef-2025/test_soundscapes'
# test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
# test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
# # test_audio_dir = '/kaggle/input/birdclef-2025/train_soundscapes/'
# file_list = [f for f in sorted(os.listdir(test_audio_dir))]
# ogg_files = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
# # file_list = file_list[:10]
# debug = False
# print(len(test_soundscapes))
# if len(ogg_files) == 0:
#     debug = True
#     debug_st_num = 1
#     debug_num = 1
#     test_audio_dir = '/kaggle/input/birdclef-2025/train_soundscapes/'
#     file_list = [f for f in sorted(os.listdir(test_audio_dir))]
#     file_list = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
#     ogg_files = file_list[debug_st_num:debug_st_num+debug_num]
# test_data = []

# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['primary_label'])
# actual_class_names = label_encoder.classes_

# print(ogg_files)


# for file in ogg_files:
#     print(file)
#     try:
#         audio,_ = librosa.load(f"{test_audio_dir}{file}.ogg", duration=10)
#         mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLING_RATE, n_mfcc=40).T,axis=0)
#         mfccs_1 = [0]*40
#     except Exception(e):
#         mfccs_1 = [0]*40
    
#     test_data.append(mfccs_1)
#     # predictions = model_1.predict(mfccs)4

# X_test = np.array(test_data)
# print(X_test.shape)

# # X_test = X_test.reshape(X_test.shape[0], -1)
# # try:
# probabilities = model.predict(X_test)


# # Create a DataFrame with file names and predicted probabilities for each class
# df_predictions = pd.DataFrame(probabilities, columns=actual_class_names)

# df_predictions['row_id'] = ogg_files

# df_predictions = df_predictions[['row_id'] + [col for col in df_predictions.columns if col != 'row_id']]

# print("Saving Submission File")
# df_predictions.to_csv("/kaggle/working/submission.csv", index=False)
# df_predictions.to_csv("submission.csv", index=False)
# print("Submission file saved successfully")
# # except Exception as e:
# #     print(e)
# #     print("No Test sound files found")
# #     columns = ["row_id"] + actual_class_names.tolist()
# #     df_predictions = pd.DataFrame(columns=columns)
    
# #     df_predictions.to_csv("/kaggle/working/submission.csv", index=False)
# #     df_predictions.to_csv("submission.csv", index=False)




SAMPLING_RATE = 32000
test_audio_dir = '/kaggle/input/birdclef-2025/test_soundscapes/'
file_list = [f for f in sorted(os.listdir(test_audio_dir))]
ogg_files = [file.split('.')[0] for file in file_list if file.endswith('.ogg')]
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]


# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['primary_label'])
# actual_class_names = label_encoder.classes_

# # test_data = [[0]*40]
# # X_test = np.array(test_data)
# # probabilities = [[1]*206]

# probabilities = []
# # test_soundscapes = test_soundscapes[1:10]
# for file in test_soundscapes:
#     audio,_ = librosa.load(file, duration=10)
#     print(file)
#     # data = [0.81]*206
#     data = np.random.rand(206)
#     probabilities.append(data)


# df_predictions = pd.DataFrame(probabilities, columns=actual_class_names)
# df_predictions['row_id'] = ogg_files
# df_predictions = df_predictions[['row_id'] + [col for col in df_predictions.columns if col != 'row_id']]

# print("Saving Submission File")
# # df_predictions.to_csv("submission.csv", index=False)

# df_predictions.to_csv('submission.csv', index=False)
# df_predictions.head()

# test_data = []
# if len(test_soundscapes) !=0:
#     for file in test_soundscapes:
#         mfccs = [0]*40
#         test_data.append(mfccs)
# else:
#     test_data = [[0]*40]
# X_test = np.array(test_data)

# probabilities = model.predict(X_test)

# df_predictions = pd.DataFrame(probabilities, columns=actual_class_names)
# if len(test_soundscapes) !=0: 
#     df_predictions['row_id'] = ogg_files
# else:
#     df_predictions['row_id'] = ['test']

# df_predictions = df_predictions[['row_id'] + [col for col in df_predictions.columns if col != 'row_id']]

# print("Saving Submission File")
# df_predictions.to_csv("submission.csv", index=False)


class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
predictions = pd.DataFrame(columns=['row_id'] + class_labels)
test_soundscapes = test_soundscapes
for soundscape in test_soundscapes:
    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(chunk)
     
    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):
        try:
            # Get row id  (soundscape id + end time of 5s chunk)      
            row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
            # mfccs = np.mean(librosa.feature.mfcc(y=chunk, sr=SAMPLING_RATE, n_mfcc=40).T,axis=0)
            # Make prediction (let's use random scores for now)
            # scores = model.predict...
            # scores = np.random.rand(len(class_labels))
            scores = model.predict(np.array(mfccs))
        except:
            scores = np.random.rand(len(class_labels))
        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
        
# Save prediction as csv
predictions.to_csv('submission.csv', index=False)
predictions.head()


predictions.shape




