import pandas as pd
import numpy as np

# Load the training data.The notebook was written on kaggle thus the data path has been used like this
train_df = pd.read_csv('/kaggle/input/beginners-hypothesis-25/BH25/Training_Data/train.csv')
train_df = train_df[:5000]#we train data on half of the dataset due to accelerator and time limitations


import cv2 #to extract the videos into frames
#a function to convert a vedio into frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()#ret for availability of frame and and frame to store the frame
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))#standardization
        frames.append(frame)
    cap.release()
    frames = np.array(frames)  # Shape: (20, 64, 64, 3)
    return frames.flatten()  # Flattenning to 1D array
    
video_features = []
for idx, row in train_df.iterrows():#running the above function for every training video
    video_path = f"/kaggle/input/beginners-hypothesis-25/BH25/Training_Data/Train_Videos/{row['video_id']}.mp4"
    video_features.append(process_video(video_path))

video_features = np.array(video_features)


from sklearn.preprocessing import LabelEncoder
# Encoding categorical attributes using the Label Encoder
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
# Spliting data for training and validation
X_train_speed, X_val_speed, y_train_speed, y_val_speed = train_test_split(video_features, y_speed, test_size=0.2, random_state=42)
X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(video_features, y_categorical, test_size=0.2, random_state=42)
X_train_summary, X_val_summary, y_train_summary, y_val_summary = train_test_split(video_features, y_summary, test_size=0.2, random_state=42)


from sklearn.svm import SVR #using svm for our model


# Training SVM regression models for speed
model_continuous = SVR(kernel = 'linear', C = 1)#we use linear because through training on previous models like decision tree,xgboost,linear regression,ridge,lasso,we find that data is more accurate for linear. C=1 is a small value of C to allow a certain value of freedom
model_continuous.fit(X_train_speed, y_train_speed)


from sklearn.svm import SVC#using svm classifier for classification tyoe of data

models_categorical = {}
for col in categorical_columns:
     model=SVC(kernel = 'linear', C = 1)
     model.fit(X_train_cat, y_train_cat[col])
     models_categorical[col] = model


# Training for video_summary using separate models for x and y here

x_values, y_values = zip(*y_train_summary)

model_summary_x = SVR(kernel = 'linear', C = 1)
model_summary_x.fit(X_train_summary, x_values)

model_summary_y =SVR(kernel = 'linear', C = 1)
model_summary_y.fit(X_train_summary, y_values)


import os

test_folder = '/kaggle/input/beginners-hypothesis-25/BH25/Testing_Data/'

test_features = []
video_ids = []

# using the above created process video to prepare the test data

for video_id in sorted(
    [f for f in os.listdir(test_folder) if f.endswith(".mp4")], key=lambda x: int(x.split('.')[0])):
    video_path = os.path.join(test_folder, video_id)
    video_ids.append(video_id.split('.')[0]) 
    test_features.append(process_video(video_path))


test_features = np.array(test_features)  # Shape: (num_test_samples, 20*64*64*3)


#predicting our regression datas
y_test_speed = model_continuous.predict(test_features)
y_test_summary_x = model_summary_x.predict(test_features)
y_test_summary_y = model_summary_y.predict(test_features)


y_categorical = {}
#predicting our classification data
for col in categorical_columns:
    y_categorical[col] = models_categorical[col].predict(test_features)


# decoding the label encoding to prepare our csv submission file

decoded_categorical = {}
for col, predictions in y_categorical.items():
    rounded_predictions = np.rint(predictions).astype(int)#converting the predictions into rounded integer values
    valid_classes = range(len(label_encoders[col].classes_))#finding out the total number of categories in various types of categorial data for example no of elements in the element
    rounded_predictions = np.clip(rounded_predictions, min(valid_classes), max(valid_classes))#filtering our data for any out of range output
    decoded_categorical[col] = label_encoders[col].inverse_transform(rounded_predictions)#reversing our encoding


speed_predictions = y_test_speed.flatten()  # Ensuring that we have a 1d array firr regression values
summary_x_predictions = y_test_summary_x.flatten()
summary_y_predictions = y_test_summary_y.flatten()


output_df = pd.DataFrame({
    'video_id': video_ids,                      
    'element': decoded_categorical['element'],  
    'motion': decoded_categorical['motion'],
    'power': decoded_categorical['power'],
    'speed': speed_predictions,               
    'video_summary': list(zip(summary_x_predictions, summary_y_predictions)) 
})


output_df.to_csv("submission.csv", index=False)

# This is the submission file to be submitted

