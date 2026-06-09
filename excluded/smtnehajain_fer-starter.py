# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





import pandas as pd
df=pd.read_csv('/kaggle/input/fer2013/fer2013.csv')
print(df.head()) 
df.to_csv("/kaggle/working/my_output.csv")
print("successfully save")
        


df.shape

# Columns के नाम
df.columns

# Missing values चेक करना
df.isnull().sum()

# कुछ basic stats
df.describe()


emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
print("emotion",emotion_labels)



import pandas as pd

# Emotion label mapping
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


label_df = pd.DataFrame(list(emotion_labels.items()), columns=['Label Code', 'Emotion']) 
label_df







df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')


emotion_counts = df['emotion'].value_counts().sort_index()
emotion_names = [emotion_labels[i] for i in emotion_counts.index]


count_df = pd.DataFrame({
    'Label Code': emotion_counts.index,
    'Emotion': emotion_names,
    'Count': emotion_counts.values
})

count_df


import numpy as np
import matplotlib.pyplot as plt

row = df.iloc[0]


pixels = np.array(row['pixels'].split(), dtype='float32')
image = pixels.reshape(48, 48)  


plt.imshow(image, cmap='gray')
plt.title(emotion_labels[row['emotion']])
plt.axis('off')
plt.show()



train_data = df[df['Usage'] == 'Training']
val_data   = df[df['Usage'] == 'PublicTest']
test_data  = df[df['Usage'] == 'PrivateTest']


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')


emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

plt.figure(figsize=(10, 10))

for i in range(6):
    row = df.iloc[i]
    
 
    pixels = np.fromstring(row['pixels'], sep=' ', dtype='float32')
    image = pixels.reshape(48, 48)
    
   
    plt.subplot(2, 3, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(emotion_labels.get(row['emotion'], "Unknown"))
    plt.axis('off')

plt.tight_layout()
plt.show()
plt. savefig("/kaggle/working/graph.png")
plt. close()



train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)

plt.figure(figsize=(10, 10))
for i in range(24):
    row = train_df.iloc[i]
    pixels = np.fromstring(row['pixels'], sep=' ', dtype='float32')
    image = pixels.reshape(48, 48)
    
    plt.subplot(6, 4, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(emotion_labels.get(row['emotion'], "Unknown"))
    plt.axis('off')

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')

# Emotion label mapping
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Emotion label counts
emotion_counts = df['emotion'].value_counts().sort_index()
emotion_names = [emotion_labels[i] for i in emotion_counts.index]


plt.figure(figsize=(8, 4))
plt.bar(emotion_names, emotion_counts.values, color='pink')
plt.title('Emotion Distribution in FER2013')
plt.xlabel('Emotions')
plt.ylabel('Number of Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt. savefig("/kaggle/working/graph.png")
plt. close()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')

# Emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}
# Preprocess images
X = []
y = []

for i in range(len(df)):
    pixels = np.fromstring(df['pixels'][i], sep=' ', dtype='float32')
    image = pixels.reshape(48, 48, 1) / 255.0
    X.append(image)
    y.append(df['emotion'][i])
    
X = np.array(X)
y = to_categorical(y, num_classes=7)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64)
# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your test image
img = cv2.imread('/kaggle/input/fer2013/fer2013.csv')  # change path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (48, 48))
    face_normalized = face_resized.astype('float32') / 255.0
    face_input = np.expand_dims(face_normalized.reshape(48, 48, 1), axis=0)
    
    prediction = model.predict(face_input)
    emotion = emotion_labels[np.argmax(prediction)]

    # Draw box and label
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Show image
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Face Detection + Emotion')
plt.show()




data = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')

# Show basic info
print(data.shape)
print(data['emotion'].value_counts())



model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuary'])
model.summary()



import pandas as pd
import matplotlib.pyplot as plt

# Dataset load karo
df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')

# Emotion labels mapping
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Value counts nikaalo (i.e., har emotion ka count)
emotion_counts = df['emotion'].value_counts().sort_index()
labels = [emotion_labels[i] for i in emotion_counts.index]

# Pie chart banao
plt.figure(figsize=(4,4))
plt.pie(emotion_counts, labels=labels, autopct='%1.1f%%', startangle=100)
plt.title('Emotion Distribution in FER2013 Dataset')
plt.axis('equal')  # Circle shape
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Dataset load karo
df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')

# Emotion labels mapping
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Emotion label counts
emotion_counts = df['emotion'].value_counts().sort_index()
labels = [emotion_labels[i] for i in emotion_counts.index]

# Line chart banao
plt.figure(figsize=(6,2))
plt.plot(labels, emotion_counts, marker='o', linestyle='-', color='blue')
plt.title('Emotion Distribution (Line Chart)')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


y_pred = model.predict(X_test)

# Convert one-hot to label indices
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_labels, y_pred_labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




cm = confusion_matrix(y_test_labels, y_pred_labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("Classification Report:\n")
print(classification_report(y_test_labels, y_pred_labels, target_names=emotion_labels))



import pandas as pd

# Dataset path
df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')

# Use only 3 emotions for simplicity
df = df[df['emotion'].isin([0, 3, 4])]  # 0: angry, 3: happy, 4: sad
emotion_map = {0: 'Angry', 3: 'Happy', 4: 'Sad'}
df['emotion_label'] = df['emotion'].map(emotion_map)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Convert pixels string to numpy arrays
X = np.array([np.fromstring(pix, sep=' ') for pix in df['pixels']])
y = df['emotion']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1. Load CSV
# -------------------------
data = pd.read_csv("/kaggle/input/fer2013/fer2013.csv")
print("Dataset loaded:", data.shape)
print(data.head())

# -------------------------
# 2. Convert Pixels to Numpy Array
# -------------------------
# har row ke 'pixels' ko array me convert
X = np.array([np.fromstring(p, dtype=int, sep=" ") for p in data['pixels']])
print("Pixels shape (flat):", X.shape)

# -------------------------
# 3. Reshape into Images (48x48 grayscale)
# -------------------------
X = X.reshape(-1, 48, 48, 1)
X = X / 255.0   # normalize [0,1]
print("Image shape:", X.shape)

# -------------------------
# 4. Labels (Emotions)
# -------------------------
y = to_categorical(data['emotion'], num_classes=7)
print("Labels shape:", y.shape)

# -------------------------
# 5. Split into Train/Val sets
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set:", X_train.shape, y_train.shape)
print("Val set:", X_val.shape, y_val.shape)
import numpy as np

# Save data into one compressed file
np.savez_compressed(
    "fer2013_preprocessed.npz", 
    X_train=X_train, 
    y_train=y_train, 
    X_val=X_val, 
    y_val=y_val
)

print("Preprocessed dataset saved to fer2013_preprocessed.npz ✅")
data = np.load("fer2013_preprocessed.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

print("Loaded preprocessed data:")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)




import numpy as np

# Load preprocessed dataset
data = np.load("fer2013_preprocessed.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


import numpy as np

# Preprocessed dataset load karo
data = np.load("fer2013_preprocessed.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
model.save("emotion_cnn.h5")
model = load_model("emotion_cnn.h5")
model = load_model("/kaggle/working/emotion_cnn.h5")



import cv2
import os

print(os.listdir("/kaggle/working"))  # चेक करो test.jpg दिख रहा है या नहीं

img = cv2.imread("/kaggle/working/test.jpg")
if img is None:
    print("Image not found!")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)   # अगर GUI नहीं है तो imshow काम नहीं करेगा
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import cv2
import numpy as np
from keras.models import load_model

# Load trained CNN model
model = load_model("emotion_cnn.h5")
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Test image (आप कोई भी image अपलोड करके path यहाँ डालें)
img = cv2.imread("test.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi_gray, (48,48))
    roi = roi.astype("float")/255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    preds = model.predict(roi)
    label = emotion_labels[np.argmax(preds)]

    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.putText(img, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Save output file instead of imshow
cv2.imwrite("output.jpg", img)
print("✅ Result saved as output.jpg")
cap = cv2.VideoCapture("input_video.mp4")
out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi_gray, (48,48))
        roi = roi.astype("float")/255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi)
        label = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    out.write(frame)

cap.release()
out.release()
print("✅ Video saved as output_video.avi")


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 2: Load Dataset
data = pd.read_csv("/kaggle/input/fer2013/fer2013.csv")

# Convert pixel data to numpy arrays
X = []
for i in range(len(data)):
    pixels = np.array(data['pixels'][i].split(), dtype='float32')
    image = pixels.reshape(48,48,1)
    X.append(image)

X = np.array(X)
y = to_categorical(data['emotion'], num_classes=7)

# Normalize
X = X / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Train CNN
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

# Step 5: Viola-Jones Face Detection (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Example of detecting face and predicting emotion
def detect_and_predict(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48,48))
        roi_resized = roi_resized.reshape(1,48,48,1) / 255.0

        prediction = model.predict(roi_resized)
        emotion = np.argmax(prediction)

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img, str(emotion), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()






import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 2: Load Dataset (MNIST digits)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to (28,28,1) because CNN expects 3D input
X_train = X_train.reshape(-1,28,28,1).astype('float32') / 255.0
X_test  = X_test.reshape(-1,28,28,1).astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Step 3: CNN Model
model = Sequential([
    Conv2D(128, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax')
])

# Step 4: Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# Step 6: Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
from tensorflow.keras.utils import plot_model

# Diagram save as PNG
plot_model(model, to_file="model_design.png", show_shapes=True, show_layer_names=True)


history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


import pandas as pd

pd.DataFrame(history.history).to_csv("training_results.csv", index=False)


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


data = pd.read_csv("/kaggle/input/fer2013/fer2013.csv")

# Prepare data
X = []
y = []

for i, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype="float32")
    image = pixels.reshape(48, 48, 1)   # FER2013 is 48x48 grayscale
    X.append(image)
    y.append(row['emotion'])

X = np.array(X) / 255.0   # Normalize [0,1]
y = to_categorical(y, num_classes=7)

print("Dataset shape:", X.shape, y.shape)   # Expected: (35887, 48, 48, 1) , (35887, 7)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64
)


model.save("fer2013_cnn.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

cap = cv2.VideoCapture(0)  # open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi_gray = roi_gray.reshape(1,48,48,1) / 255.0

        preds = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0,255,0), 2)

    cv2.imshow("Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Model prediction on test set
y_pred_probs = model.predict(X_test)                # probabilities
y_pred_classes = np.argmax(y_pred_probs, axis=1)    # predicted labels
y_true = np.argmax(y_test, axis=1)                  # true labels

# 2. Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# 3. Emotion labels for FER-2013
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# 4. Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - FER-2013")
plt.show()

# 5. Print classification report (optional for paper)
print(classification_report(y_true, y_pred_classes, target_names=emotion_labels))
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')


