
import pandas as pd
import numpy as np

# importing the file containing the features for each video.

train_df = pd.read_csv('/kaggle/input/beginners-hypothesis-25/BH25/Training_Data/train.csv')
train_df = train_df[:5000]
train_df


''' with 10000 videos , it was taking a lot of time to train , so started with 500 videos then 1000 videos , 3000 videos then eventualy 5000 video .
 I was not able to submit the prediction of the model using 10000 videos . so the score on the leaderboard is due to a model trained with 5000 videos only .



import os
import cv2
import numpy as np

# Define the directory containing the videos
video_dir = "/kaggle/input/beginners-hypothesis-25/BH25/Training_Data/Train_Videos"

# List all video files in the directory
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')], key=lambda x: int(x.split('.')[0]))


video_files = video_files[:5000]

# Process each video file
video_features = []  # To store processed video data

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resizing frame to 64x64
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)
    cap.release()
    return np.array(frames).flatten()  # Flatten the frames , originally it was of dimension 20*64*64*3

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    features = process_video(video_path)
    video_features.append(features)

# Convert video features to a NumPy array
video_features = np.array(video_features)
video_features



from sklearn.preprocessing import LabelEncoder # using label encoders to convert categorial variables to numeric data which the model can easily evaluate. 

def encode_categorical_columns(df, columns): # defining a function to encode a categorial variable to a numeric value
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

categorical_columns = ['element', 'motion', 'power']
train_df, label_encoders = encode_categorical_columns(train_df, categorical_columns) 
train_df


from sklearn.model_selection import train_test_split #spliting the data to make samples separately for traing and testing . Evaluating with new data gives a better realisation of how the model is performing.


y_speed = train_df[['speed']]
y_summary = train_df['video_summary'].apply(lambda x: eval(x))  # Convert string tuples to actual tuples

y_categorical = train_df[categorical_columns]


X_train_speed, X_val_speed, y_train_speed, y_val_speed = train_test_split(video_features, y_speed, test_size=0.2, random_state=42)
X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(video_features, y_categorical, test_size=0.2, random_state=42)
X_train_summary, X_val_summary, y_train_summary, y_val_summary = train_test_split(video_features, y_summary, test_size=0.2, random_state=42)


from sklearn.svm import SVR #using support vector regression to compensate for any possible non linearity in data .


model_speed = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # Adjusting the hyperparameters to get better score
# kernel :- i chose 'rbf(radial basis function)' kernel because there can be a possibility of data having some non-linear character.
# c (regularisation parameter) :- i set this parameter to a low setting to prevent overfitting of the model and to prevent bad score due to noise in the data and obtain a smooth relation.
# epsilon :- i set it to a high value of 0.1 , beacause i wanted to neglect any errors that might have arose due to some noise in the data.

model_speed.fit(X_train_speed, y_train_speed)
# i was not able to upload the csv file resulting from svr model for speed due to running out gpu limit . the score on the leaderboard is of linear regression only .  


model_speed.score(X_val_speed , y_val_speed) 
#evaluating the score for model_speed , with linear regression it was less than 0.65 , with support vector regression non linear data can be evaluated easily , with svr was able to get scores above 0.75.


# i could also use svr here , but it takes a lot of time to train with svr , so i did not change it here .
from sklearn.linear_model import LinearRegression
x_values, y_values = zip(*y_train_summary)

model_summary_x = LinearRegression()
model_summary_x.fit(X_train_summary, x_values)

model_summary_y = LinearRegression()
model_summary_y.fit(X_train_summary, y_values)





#testing several classification methods , linear regression wont give good results because it is classification problem, random forest gave best results.
#random forest classifier is generally better than svc because it can handle higher dimensional data more easily than svc and decission tree .  


'''from sklearn.svm import SVC

models_categorical = {}
for col in categorical_columns:
    model = SVC()
    model.fit(X_train_cat, y_train_cat[col])
    models_categorical[col] = model'''
from sklearn.ensemble import RandomForestClassifier
models_categorical = {}
for col in categorical_columns:   # the model for each feature gets appended in the dictionary in each iteration
    model = RandomForestClassifier()
    model.fit(X_train_cat, y_train_cat[col])
    models_categorical[col] = model
'''from sklearn import tree

models_categorical = {}
for col in categorical_columns:
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_cat, y_train_cat[col])
    models_categorical[col] = model'''
'''from sklearn.linear_model import LogisticRegression




models_categorical['element'].score(X_val_cat , y_val_cat.element)
models_categorical['motion'].score(X_val_cat , y_val_cat.motion)
models_categorical['power'].score(X_val_cat , y_val_cat.power)
# checking the score for categorial model , 0.803 for element , above 0.4 for power and 1.0 for motion


import os

test_folder = '/kaggle/input/beginners-hypothesis-25/BH25/Testing_Data'

test_features = []
video_ids = []
# importing test videos and making the array named test_features which stores the data of each video

for video_id in sorted(
    [f for f in os.listdir(test_folder) if f.endswith(".mp4")], key=lambda x: int(x.split('.')[0])):
    video_path = os.path.join(test_folder, video_id)
    video_ids.append(video_id.split('.')[0]) 
    test_features.append(process_video(video_path))

test_features = np.array(test_features)  # Shape: (num_test_samples, 20*64*64*3)


# predicting the values for speed and summary using test data 
y_test_speed = model_speed.predict(test_features)
y_test_summary_x = model_summary_x.predict(test_features)
y_test_summary_y = model_summary_y.predict(test_features)


#  predicting the values for categorical features
y_categorical = {}

for col in categorical_columns:
    y_categorical[col] = models_categorical[col].predict(test_features)


# converting the labels for the categorial variables to their actual names 
decoded_categorical = {}
for col, predictions in y_categorical.items():
    rounded_predictions = np.rint(predictions).astype(int)
    valid_classes = range(len(label_encoders[col].classes_))
    rounded_predictions = np.clip(rounded_predictions, min(valid_classes), max(valid_classes))
    decoded_categorical[col] = label_encoders[col].inverse_transform(rounded_predictions)


speed_predictions = y_test_speed.flatten()  # Ensure the shape is (num_samples,)
summary_x_predictions = y_test_summary_x.flatten()
summary_y_predictions = y_test_summary_y.flatten()


# constructing the dataframe to be exported to csv file.
output_df = pd.DataFrame({
    'video_id': video_ids,                      
    'element': decoded_categorical['element'],  
    'motion': decoded_categorical['motion'],
    'power': decoded_categorical['power'],
    'speed': speed_predictions,               
    'video_summary': list(zip(summary_x_predictions, summary_y_predictions)) 
})


output_df.to_csv("submission_10000(only for speed and summary)_12:20_am.csv", index=False) #exporting the csv file 




